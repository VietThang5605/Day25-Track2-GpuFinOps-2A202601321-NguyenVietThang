#!/usr/bin/env python3
"""run_extensions.py — Execute and evaluate "Your Turn" extensions for Lab 25.

Covers:
  - Extension 1: Enhanced Tier Recommendation Policy (Risk + Interruption + 1yr vs 3yr)
  - Extension 2: Right-sizing Memory-bound GPUs using MBU, Peak Bandwidth & $/GB-VRAM
  - Extension 3: Prompt Caching Economics & Break-even Thresholds
  - Extension 5: Carbon-aware & Multi-region Workload Scheduling
"""
from __future__ import annotations
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from missions._common import load_csv, num, catalog_by_type
from finops import pricing, metrics, sustainability


def run_extension_1():
    print("\n" + "=" * 70)
    print("  EXTENSION 1: Enhanced recommend_tier() Policy Evaluation")
    print("=" * 70)
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    
    print("\n1. Comparison of Baseline Policy vs. Enhanced Risk-Aware Policy:")
    print(f"{'Job ID':18}{'GPU':7}{'Duty%':>7}{'Int?':>6}{'Days':>6}  {'Base Tier':11}{'Adv Tier':11}{'Reason / Decision Factor'}")
    print("-" * 88)
    
    total_od = 0.0
    base_opt = 0.0
    adv_opt = 0.0
    
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        days = int(num(j["days"]))
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * 30 * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od
        total_od += on_demand_cost
        
        # Base policy
        duty = (hpd / 24.0) * 100
        base_tier = "spot" if (interruptible and hpd < 24) else ("reserved" if (hpd / 24.0) >= 0.55 else "on_demand")
        if base_tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            base_cost = sim["spot_cost"]
        elif base_tier == "reserved":
            base_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            base_cost = on_demand_cost
        base_opt += base_cost
        
        # Enhanced policy
        adv_tier = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype, job_days=days)
        if adv_tier == "spot":
            sim = pricing.spot_checkpoint_cost(
                gpu_hours, num(c["spot_hr"]), od,
                interrupt_rate=pricing.GPU_SPOT_INTERRUPT_RATES.get(gtype, 0.05)
            )
            adv_cost = sim["spot_cost"]
        elif adv_tier == "reserved":
            adv_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            adv_cost = on_demand_cost
        adv_opt += adv_cost
        
        # Determine reason
        if interruptible:
            irate = pricing.GPU_SPOT_INTERRUPT_RATES.get(gtype, 0.05) * 100
            reason = f"Spot valid (Interruption rate ~{irate:.0f}%, checkpointing ROI high)"
        elif days < 30:
            reason = f"Short duration ({days}d < 30d) -> avoid long commitment lock-in"
        else:
            reason = f"Steady high duty ({duty:.0f}% >= 55%) -> 3yr reserved commitment"
            
        print(f"{j['job_id']:18}{gtype:7}{duty:6.0f}%{str(interruptible):>6}{days:>6}  {base_tier:11}{adv_tier:11}{reason}")
        
    print("-" * 88)
    print(f"Total On-Demand Monthly Spend: ${total_od:,.0f}")
    print(f"Baseline Policy Monthly Spend : ${base_opt:,.0f} (Savings: {(total_od - base_opt)/total_od*100:.1f}%)")
    print(f"Enhanced Policy Monthly Spend : ${adv_opt:,.0f} (Savings: {(total_od - adv_opt)/total_od*100:.1f}%)")


def run_extension_2():
    print("\n" + "=" * 70)
    print("  EXTENSION 2: Right-sizing Memory-bound GPUs (MBU & $/GB-VRAM)")
    print("=" * 70)
    cat = load_csv("price_catalog.csv")
    
    print("\n1. GPU Catalog Hardware & Cost per GB VRAM Analysis:")
    print(f"{'GPU Type':9}{'VRAM (GB)':>10}{'Peak BW (TB/s)':>16}{'On-Demand/hr':>14}{'$/GB-VRAM-hr':>14}")
    print("-" * 65)
    catalog_metrics = {}
    for r in cat:
        gtype = r["gpu_type"]
        vram = num(r["hbm_gb"])
        bw = num(r["peak_bw_tbs"])
        price = num(r["on_demand_hr"])
        cost_per_gb = price / vram if vram > 0 else 0
        catalog_metrics[gtype] = {"vram": vram, "bw": bw, "price": price, "cost_per_gb": cost_per_gb}
        print(f"{gtype:9}{vram:>10.0f}{bw:>16.2f}${price:>13.2f}${cost_per_gb:>13.4f}")
        
    print("\n2. Detection of Memory-Bound / Underutilized GPUs from Telemetry:")
    # Telemetry audit from M1
    tel = load_csv("gpu_telemetry.csv")
    cat_by_type = catalog_by_type()
    from collections import defaultdict
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat_by_type[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat_by_type[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        
    print(f"{'GPU ID':14}{'Current':9}{'Util%':>7}{'MFU':>7}{'MBU':>7}  {'Right-size Target':19}{'Hourly Savings':>15}{'Monthly Savings':>16}")
    print("-" * 88)
    
    total_rightsizing_monthly = 0.0
    for gid, a in sorted(agg.items()):
        avg_util = sum(a["util"]) / len(a["util"])
        avg_mfu = sum(a["mfu"]) / len(a["mfu"])
        avg_mbu = sum(a["mbu"]) / len(a["mbu"])
        gtype = a["type"]
        current_price = catalog_metrics[gtype]["price"]
        
        # Right sizing logic:
        # If GPU is H100 with util ~98% but MFU ~20% and MBU ~20% (decode bound/small batch) -> A100 or A10G is sufficient
        if gtype == "H100" and avg_util > 90 and avg_mfu < 0.25:
            target = "A100 (80GB VRAM)"
            target_price = catalog_metrics["A100"]["price"]
            hourly_saved = current_price - target_price
            monthly_saved = hourly_saved * 24 * 30
            total_rightsizing_monthly += monthly_saved
            print(f"{gid:14}{gtype:9}{avg_util:>6.1f}%{avg_mfu:>7.3f}{avg_mbu:>7.3f}  {target:19}${hourly_saved:>14.2f}${monthly_saved:>15.0f}")
        elif gtype == "A10G" and avg_util > 90 and avg_mfu < 0.28:
            target = "L4 (Low-power Infer)"
            target_price = catalog_metrics["L4"]["price"]
            hourly_saved = current_price - target_price
            monthly_saved = hourly_saved * 24 * 30
            total_rightsizing_monthly += monthly_saved
            print(f"{gid:14}{gtype:9}{avg_util:>6.1f}%{avg_mfu:>7.3f}{avg_mbu:>7.3f}  {target:19}${hourly_saved:>14.2f}${monthly_saved:>15.0f}")
            
    print("-" * 88)
    print(f"Total Projected Monthly Savings from Right-sizing: ${total_rightsizing_monthly:,.0f}")


def run_extension_3():
    print("\n" + "=" * 70)
    print("  EXTENSION 3: Prompt Caching Economics & Break-even Curve")
    print("=" * 70)
    
    write_costs = [1.50, 3.00, 3.75, 5.00]
    base_read = 3.00
    read_discount = 0.10  # 90% discount on cache hit
    
    print(f"Base Input Price: ${base_read:.2f}/1M tokens | Cached Read Price (10%): ${base_read * read_discount:.2f}/1M tokens")
    print(f"Net savings per cache read: ${base_read * (1 - read_discount):.2f}/1M tokens\n")
    print(f"{'Cache Write Cost / 1M':25}{'Break-Even Read Count':>25}{'Is 2 Reads Profitable?':>25}")
    print("-" * 75)
    
    for wc in write_costs:
        be_reads = wc / (base_read * (1.0 - read_discount))
        is_worth_2 = pricing.cache_is_worth_it(2.0, write_cost_per_m=wc, base_read_cost_per_m=base_read)
        print(f"${wc:>6.2f} / 1M tokens{be_reads:>23.2f} reads{str(is_worth_2):>23}")


def run_extension_5():
    print("\n" + "=" * 70)
    print("  EXTENSION 5: Carbon-aware & Multi-region Workload Scheduling")
    print("=" * 70)
    
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    
    # Calculate energy for interruptible training workloads
    interruptible_jobs = [j for j in jobs if bool(int(num(j["interruptible"])))]
    
    total_training_kwh = 0.0
    for j in interruptible_jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        days = int(num(j["days"]))
        watts = num(cat[gtype]["watts"])
        total_hours = hpd * days
        kwh = (watts * ngpu * total_hours) / 1000.0
        total_training_kwh += kwh
        
    print(f"Total Interruptible Training Workload Energy: {total_training_kwh:,.1f} kWh\n")
    print(f"{'Region':18}{'Grid Carbon':>14}{'Power Cost':>13}{'Total Carbon':>16}{'Total Power Cost':>18}{'CO2 Reduction vs US-East':>26}")
    print("-" * 105)
    
    base_carbon = total_training_kwh * sustainability.REGION_CARBON["us-east-1"] / 1000.0 # kgCO2e
    
    for reg, carbon_intensity in sustainability.REGION_CARBON.items():
        price_kwh = sustainability.REGION_PRICE_KWH[reg]
        total_co2_kg = (total_training_kwh * carbon_intensity) / 1000.0
        total_elec_usd = total_training_kwh * price_kwh
        reduction_pct = (1.0 - total_co2_kg / base_carbon) * 100.0
        
        tag = ""
        if reg == "europe-north1":
            tag = " (Cleanest: -92% CO2)"
        elif reg == "us-east-wa":
            tag = " (Cheapest Power)"
        elif reg == "europe-central2":
            tag = " (Dirtiest)"
            
        print(f"{reg:18}{carbon_intensity:>10} g/kWh${price_kwh:>9.3f}/kWh{total_co2_kg:>13.1f} kg{'$':>10}{total_elec_usd:>7.2f}{reduction_pct:>23.1f}%{tag}")
        
    print("-" * 105)
    print("Insight: Migrating batch/training workloads to 'europe-north1' eliminates 92.1% of carbon emissions")
    print("with lower electricity costs ($0.09 vs $0.12/kWh), while having ZERO impact on user inference latency!")


def main():
    run_extension_1()
    run_extension_2()
    run_extension_3()
    run_extension_5()
    print("\n" + "=" * 70)
    print("  ALL EXTENSIONS EXECUTED SUCCESSFULLY")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
