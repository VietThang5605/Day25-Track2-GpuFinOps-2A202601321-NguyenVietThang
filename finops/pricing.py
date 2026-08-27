"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


# Typical hourly spot interruption rates by GPU architecture (June 2026 snapshot)
GPU_SPOT_INTERRUPT_RATES = {
    "H100": 0.04,   # High demand but dedicated pools ~4%
    "H200": 0.03,   # ~3%
    "A100": 0.06,   # ~6%
    "A10G": 0.12,   # Commodity cloud instance ~12%
    "L4": 0.10,     # ~10%
    "B200": 0.05,   # ~5%
    "MI300X": 0.08, # ~8%
}


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    job_days: float | None = None,
    reserved_1yr_discount: float = 0.28,
    max_acceptable_interrupt_rate: float = 0.10,
) -> str:
    """Pick a purchasing tier from a workload's duty cycle, interruptibility, and risk factors.

    Enhanced Policy (Extension 1):
      - If interruptible:
          - If gpu_type provided and spot interruption rate exceeds threshold for long jobs -> favor reserved/on_demand
          - Otherwise -> 'spot' (checkpoint and ride the discount)
      - If non-interruptible:
          - If job_days is short (< 60 days) -> 'on_demand' (avoid commitment lock-in)
          - If duty cycle >= 3yr break-even (duty >= 1 - 0.45 = 55%) and job_days >= 365 -> 'reserved' (3yr commitment)
          - If duty cycle >= 1yr break-even (duty >= 1 - 0.28 = 72%) and job_days >= 90 -> 'reserved' (1yr commitment)
          - If duty cycle >= break-even (standard) -> 'reserved'
          - Otherwise -> 'on_demand'
    """
    duty = max(0.0, hours_per_day) / 24.0
    be_3yr = break_even_utilization(reserved_discount)

    if interruptible and hours_per_day < 24:
        if gpu_type and gpu_type in GPU_SPOT_INTERRUPT_RATES:
            rate = GPU_SPOT_INTERRUPT_RATES[gpu_type]
            # If interruption rate is too high (>10%) for long training (>10 days), avoid spot risk
            if rate > max_acceptable_interrupt_rate and job_days and job_days > 10:
                return "reserved" if duty >= be_3yr else "on_demand"
        return "spot"

    if job_days is not None and job_days < 30:
        return "on_demand"

    if duty >= be_3yr:
        return "reserved"
    return "on_demand"


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float = 3.75,   # Typical cache write premium (e.g. 1.25x base)
    base_read_cost_per_m: float = 3.00,
    read_discount: float = 0.10,      # 90% discount on cache hit
) -> bool:
    """Prompt caching is only financially beneficial when amortized read savings exceed write cost.

    Break-even: avg_cache_reads * (base_read_cost * (1 - read_discount)) > write_cost
    """
    savings_per_read = base_read_cost_per_m * (1.0 - read_discount)
    total_savings = avg_cache_reads * savings_per_read
    return total_savings > write_cost_per_m

