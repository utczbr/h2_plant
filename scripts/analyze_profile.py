"""
Analyze profile stats.
"""
import pstats
import sys

stats = pstats.Stats('profile.stats')
stats.sort_stats('cumulative')

print("=" * 60)
print("TOP 50 FUNCTIONS BY CUMULATIVE TIME")
print("=" * 60)
stats.print_stats(50)

print("\n" + "=" * 60)
print("TOP 50 FUNCTIONS BY TOTAL TIME (SELF)")
print("=" * 60)
stats.sort_stats('tottime')
stats.print_stats(50)

print("\n" + "=" * 60)
print("MIXTURE THERMODYNAMICS CALLS")
print("=" * 60)
stats.print_stats('mixture_thermodynamics')
