from collections import defaultdict

def groupAnagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))  # 'eat' → ('a','e','t')
        groups[key].append(s)
    return list(groups.values())

print(groupAnagrams(["eat","tea","tan","ate","nat","bat"]))
# [['eat','tea','ate'], ['tan','nat'], ['bat']]
