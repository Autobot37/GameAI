import pstats

# Load the profile data
p = pstats.Stats('out.prof')

# Sort by cumulative time and print the top 10 results
p.sort_stats('tottime').print_stats(10)

