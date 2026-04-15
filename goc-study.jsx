import { useState } from "react";

const problems = [
  // ===== BFS/DFS =====
  {
    id: 1,
    category: "BFS/DFS",
    difficulty: "Medium",
    title: "遅延プロジェクトの検出",
    subtitle: "GOC 2019 Q1 類似",
    description: `依存グラフが与えられる。各ノードはプロジェクト、有向エッジは「依存関係」を表す。
サイクルに含まれるプロジェクト（循環依存で永遠に完了できないもの）を全て返せ。`,
    example: `入力: n=4, edges=[[0,1],[1,2],[2,0],[3,1]]
出力: [0, 1, 2]  # 3はサイクル外`,
    hint: "トポロジカルソート（Kahn's algorithm）でサイクル検出。in-degreeが0のノードをキューに入れ、除去できなかったノードがサイクル。",
    solution: `from collections import deque

def find_delayed_projects(n, edges):
    graph = [[] for _ in range(n)]
    in_degree = [0] * n
    
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    
    # in_degree==0 のノードからBFS
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    visited = set()
    
    while queue:
        node = queue.popleft()
        visited.add(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # visitedされなかった = サイクル内
    return sorted([i for i in range(n) if i not in visited])

# Test
print(find_delayed_projects(4, [[0,1],[1,2],[2,0],[3,1]]))
# [0, 1, 2]`,
    complexity: "時間: O(V+E)　空間: O(V+E)"
  },
  {
    id: 2,
    category: "BFS/DFS",
    difficulty: "Medium",
    title: "島の数",
    subtitle: "LeetCode 200",
    description: `'1'（陸地）と'0'（水）からなる2次元グリッドが与えられる。島（水で囲まれた陸地の塊）の数を返せ。`,
    example: `入力:
[["1","1","0","0"],
 ["1","1","0","0"],
 ["0","0","1","0"],
 ["0","0","0","1"]]
出力: 3`,
    hint: "各'1'からDFSで周囲の陸地を全て'0'に塗り潰す。DFSを呼んだ回数が答え。",
    solution: `def numIslands(grid):
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        if grid[r][c] != '1':
            return
        grid[r][c] = '0'  # 訪問済みにする
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)
                count += 1
    
    return count

grid = [["1","1","0","0"],["1","1","0","0"],
        ["0","0","1","0"],["0","0","0","1"]]
print(numIslands(grid))  # 3`,
    complexity: "時間: O(M×N)　空間: O(M×N)"
  },
  {
    id: 3,
    category: "BFS/DFS",
    difficulty: "Medium",
    title: "コース受講可能判定",
    subtitle: "LeetCode 207",
    description: `numCoursesコース（0〜numCourses-1）があり、prerequisites[i]=[a,b]は「aを受けるにはbが必要」を意味する。全コースを修了できるか判定せよ。`,
    example: `入力: numCourses=2, prerequisites=[[1,0],[0,1]]
出力: False  # 循環依存あり`,
    hint: "有向グラフのサイクル検出。DFSで各ノードに「未訪問/訪問中/訪問済」の3状態を持たせる。訪問中のノードに再訪したらサイクル。",
    solution: `def canFinish(numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    for a, b in prerequisites:
        graph[b].append(a)
    
    # 0: 未訪問, 1: 訪問中, 2: 訪問済
    state = [0] * numCourses
    
    def dfs(node):
        if state[node] == 1:  # サイクル検出
            return False
        if state[node] == 2:  # 処理済み
            return True
        
        state[node] = 1
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        state[node] = 2
        return True
    
    return all(dfs(i) for i in range(numCourses))

print(canFinish(2, [[1,0],[0,1]]))  # False
print(canFinish(2, [[1,0]]))        # True`,
    complexity: "時間: O(V+E)　空間: O(V+E)"
  },
  // ===== Sort + HashMap =====
  {
    id: 4,
    category: "ソート＋ハッシュ",
    difficulty: "Easy",
    title: "相対ソート",
    subtitle: "GOC 2019 Q2 / LeetCode 1122",
    description: `配列arr1とarr2が与えられる。arr2の要素は全てユニークでarr1に含まれる。
arr1をarr2の順番に従ってソートせよ。arr2にない要素は末尾に昇順で置く。`,
    example: `入力: arr1=[2,3,1,3,2,4,6,7,9,2,19], arr2=[2,1,4,3,9,6]
出力: [2,2,2,1,4,3,3,9,6,7,19]`,
    hint: "arr2のインデックスをキーにした辞書を作る。arr1をカスタムキー（arr2にあれば順位、なければ(大きい数, 値)）でソート。",
    solution: `def relativeSortArray(arr1, arr2):
    order = {v: i for i, v in enumerate(arr2)}
    
    # arr2にある → (0, arr2でのindex)
    # arr2にない → (1, 値) で末尾に昇順
    return sorted(arr1, key=lambda x: (0, order[x]) if x in order else (1, x))

print(relativeSortArray(
    [2,3,1,3,2,4,6,7,9,2,19],
    [2,1,4,3,9,6]
))
# [2, 2, 2, 1, 4, 3, 3, 9, 6, 7, 19]`,
    complexity: "時間: O(N log N)　空間: O(N)"
  },
  {
    id: 5,
    category: "ソート＋ハッシュ",
    difficulty: "Medium",
    title: "アナグラムのグループ化",
    subtitle: "LeetCode 49",
    description: `文字列のリストが与えられる。アナグラム同士をグループ化して返せ。`,
    example: `入力: ["eat","tea","tan","ate","nat","bat"]
出力: [["bat"],["nat","tan"],["ate","eat","tea"]]`,
    hint: "各文字列をソートしたものをキーとしてハッシュマップに入れる。sorted('eat') == sorted('tea') == 'aet'。",
    solution: `from collections import defaultdict

def groupAnagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))  # 'eat' → ('a','e','t')
        groups[key].append(s)
    return list(groups.values())

print(groupAnagrams(["eat","tea","tan","ate","nat","bat"]))
# [['eat','tea','ate'], ['tan','nat'], ['bat']]`,
    complexity: "時間: O(N·K log K)　空間: O(N·K)　※K=最長文字列長"
  },
  {
    id: 6,
    category: "ソート＋ハッシュ",
    difficulty: "Medium",
    title: "最大Bitwise ORの最小部分集合",
    subtitle: "GOC 2020 Q1 類似",
    description: `整数配列numsが与えられる。配列全体のBitwise ORと同じ値になる最小サイズの部分集合の長さを返せ。`,
    example: `入力: [3, 1]
出力: 1  # 3だけで OR=3（全体と同じ）

入力: [2, 2, 2]
出力: 1  # どれか1つで OR=2`,
    hint: "まず全体のORを計算する。次に、その値をビット1つで達成できる要素があるか確認。なければ2つ必要かチェック（総当たりOK、N≦16程度なら）。",
    solution: `def minElements(nums):
    total_or = 0
    for n in nums:
        total_or |= n
    
    # 1要素で達成できるか
    for n in nums:
        if n == total_or:
            return 1
    
    # 2要素で達成できるか
    n = len(nums)
    for i in range(n):
        for j in range(i+1, n):
            if (nums[i] | nums[j]) == total_or:
                return 2
    
    # それ以上（GOCでは通常1か2で収まる）
    return 3

print(minElements([3, 1]))     # 1
print(minElements([1, 2, 4]))  # 3
print(minElements([2, 2, 2]))  # 1`,
    complexity: "時間: O(N²)　空間: O(1)"
  },
  // ===== DP =====
  {
    id: 7,
    category: "動的計画法",
    difficulty: "Easy",
    title: "コイン両替",
    subtitle: "LeetCode 322",
    description: `硬貨の種類（coins）と目標金額（amount）が与えられる。目標金額を作るのに必要な最小硬貨枚数を返せ。不可能なら-1を返せ。`,
    example: `入力: coins=[1,5,6,9], amount=11
出力: 2  # 5+6=11`,
    hint: "dp[i] = 金額iを作る最小枚数。dp[0]=0、dp[i] = min(dp[i-c]+1) for c in coins。",
    solution: `def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

print(coinChange([1,5,6,9], 11))  # 2
print(coinChange([2], 3))          # -1`,
    complexity: "時間: O(amount × coins数)　空間: O(amount)"
  },
  {
    id: 8,
    category: "動的計画法",
    difficulty: "Easy",
    title: "最大部分配列和",
    subtitle: "LeetCode 53 (Kadane's Algorithm)",
    description: `整数配列numsが与えられる。和が最大になる連続部分配列の和を返せ。`,
    example: `入力: [-2,1,-3,4,-1,2,1,-5,4]
出力: 6  # [4,-1,2,1]`,
    hint: "Kadane's Algorithm：現在の要素を「前の最大和に足す」か「そこから始め直す」かの選択。current = max(nums[i], current + nums[i])。",
    solution: `def maxSubArray(nums):
    current = best = nums[0]
    
    for n in nums[1:]:
        current = max(n, current + n)
        best = max(best, current)
    
    return best

print(maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))  # 6
print(maxSubArray([1]))                         # 1
print(maxSubArray([-1,-2,-3]))                  # -1`,
    complexity: "時間: O(N)　空間: O(1)"
  },
  {
    id: 9,
    category: "動的計画法",
    difficulty: "Medium",
    title: "最長増加部分列",
    subtitle: "LeetCode 300",
    description: `整数配列numsが与えられる。厳密に増加する最長部分列の長さを返せ（連続でなくてよい）。`,
    example: `入力: [10,9,2,5,3,7,101,18]
出力: 4  # [2,3,7,101]`,
    hint: "dp[i] = nums[i]で終わる最長増加部分列の長さ。dp[i] = max(dp[j]+1) for j<i where nums[j]<nums[i]。",
    solution: `def lengthOfLIS(nums):
    if not nums:
        return 0
    
    dp = [1] * len(nums)
    
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

print(lengthOfLIS([10,9,2,5,3,7,101,18]))  # 4
print(lengthOfLIS([0,1,0,3,2,3]))           # 4`,
    complexity: "時間: O(N²)　空間: O(N)　※O(N log N)解法もあり"
  }
];

const categoryColors = {
  "BFS/DFS": { bg: "#0f4c35", accent: "#34d399", light: "#d1fae5" },
  "ソート＋ハッシュ": { bg: "#1e3a5f", accent: "#60a5fa", light: "#dbeafe" },
  "動的計画法": { bg: "#4a1942", accent: "#c084fc", light: "#f3e8ff" },
};

const diffBadge = {
  "Easy": { bg: "#064e3b", color: "#6ee7b7" },
  "Medium": { bg: "#78350f", color: "#fcd34d" },
  "Hard": { bg: "#7f1d1d", color: "#fca5a5" },
};

export default function App() {
  const [selected, setSelected] = useState(null);
  const [showSolution, setShowSolution] = useState(false);
  const [filter, setFilter] = useState("全て");

  const categories = ["全て", "BFS/DFS", "ソート＋ハッシュ", "動的計画法"];
  const filtered = filter === "全て" ? problems : problems.filter(p => p.category === filter);

  function openProblem(p) {
    setSelected(p);
    setShowSolution(false);
  }

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0a0a0f",
      color: "#e2e8f0",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      padding: "0",
    }}>
      {/* Header */}
      <div style={{
        background: "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)",
        padding: "28px 32px 20px",
        borderBottom: "1px solid #1e3a5f",
      }}>
        <div style={{ fontSize: "11px", color: "#60a5fa", letterSpacing: "3px", marginBottom: "6px" }}>
          GOOGLE ONLINE CHALLENGE
        </div>
        <div style={{ fontSize: "22px", fontWeight: "700", color: "#f0f9ff", letterSpacing: "-0.5px" }}>
          GOC 対策問題集
        </div>
        <div style={{ fontSize: "12px", color: "#64748b", marginTop: "4px" }}>
          9問 · BFS/DFS · ソート · DP · Python
        </div>
        {/* Filter */}
        <div style={{ display: "flex", gap: "8px", marginTop: "16px", flexWrap: "wrap" }}>
          {categories.map(c => (
            <button key={c} onClick={() => setFilter(c)} style={{
              padding: "5px 14px",
              borderRadius: "20px",
              border: filter === c ? "1px solid #60a5fa" : "1px solid #1e3a5f",
              background: filter === c ? "#1e3a5f" : "transparent",
              color: filter === c ? "#60a5fa" : "#64748b",
              fontSize: "11px",
              cursor: "pointer",
              letterSpacing: "0.5px",
              fontFamily: "inherit",
            }}>{c}</button>
          ))}
        </div>
      </div>

      <div style={{ display: "flex", height: "calc(100vh - 130px)" }}>
        {/* Problem List */}
        <div style={{
          width: selected ? "320px" : "100%",
          minWidth: "260px",
          overflowY: "auto",
          borderRight: selected ? "1px solid #1e293b" : "none",
          padding: "12px",
          transition: "width 0.2s",
        }}>
          {filtered.map(p => {
            const col = categoryColors[p.category];
            const diff = diffBadge[p.difficulty];
            const isActive = selected?.id === p.id;
            return (
              <div key={p.id} onClick={() => openProblem(p)} style={{
                padding: "14px 16px",
                marginBottom: "8px",
                borderRadius: "8px",
                border: isActive ? `1px solid ${col.accent}` : "1px solid #1e293b",
                background: isActive ? col.bg : "#111827",
                cursor: "pointer",
                transition: "all 0.15s",
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
                  <span style={{
                    fontSize: "10px", color: col.accent,
                    background: col.bg, padding: "2px 8px",
                    borderRadius: "10px", border: `1px solid ${col.accent}33`,
                  }}>{p.category}</span>
                  <span style={{
                    fontSize: "10px", color: diff.color,
                    background: diff.bg, padding: "2px 8px", borderRadius: "10px",
                  }}>{p.difficulty}</span>
                </div>
                <div style={{ fontSize: "13px", fontWeight: "600", color: isActive ? "#f0f9ff" : "#cbd5e1" }}>
                  {p.id}. {p.title}
                </div>
                <div style={{ fontSize: "11px", color: "#475569", marginTop: "2px" }}>{p.subtitle}</div>
              </div>
            );
          })}
        </div>

        {/* Problem Detail */}
        {selected && (
          <div style={{ flex: 1, overflowY: "auto", padding: "24px 28px" }}>
            {/* Back button on mobile */}
            <button onClick={() => setSelected(null)} style={{
              background: "none", border: "1px solid #1e293b", color: "#64748b",
              padding: "4px 12px", borderRadius: "6px", cursor: "pointer",
              fontSize: "11px", fontFamily: "inherit", marginBottom: "16px",
            }}>← 戻る</button>

            {/* Title */}
            <div style={{
              fontSize: "10px", color: categoryColors[selected.category].accent,
              letterSpacing: "2px", marginBottom: "6px",
            }}>{selected.category} · {selected.subtitle}</div>
            <h2 style={{ fontSize: "20px", color: "#f0f9ff", margin: "0 0 16px" }}>
              {selected.id}. {selected.title}
            </h2>

            {/* Description */}
            <Section title="問題">
              <p style={{ color: "#94a3b8", lineHeight: "1.7", fontSize: "13px", margin: 0 }}>
                {selected.description}
              </p>
            </Section>

            {/* Example */}
            <Section title="例">
              <CodeBlock code={selected.example} />
            </Section>

            {/* Hint */}
            <Section title="ヒント">
              <p style={{ color: "#94a3b8", lineHeight: "1.7", fontSize: "13px", margin: 0 }}>
                💡 {selected.hint}
              </p>
            </Section>

            {/* Complexity */}
            <Section title="計算量">
              <div style={{
                display: "inline-block", background: "#0f172a",
                border: "1px solid #1e293b", borderRadius: "6px",
                padding: "8px 14px", fontSize: "12px", color: "#7dd3fc",
              }}>
                {selected.complexity}
              </div>
            </Section>

            {/* Solution toggle */}
            <button onClick={() => setShowSolution(!showSolution)} style={{
              marginTop: "20px",
              padding: "10px 24px",
              background: showSolution ? "#1e3a5f" : "linear-gradient(135deg, #1d4ed8, #0f3460)",
              border: "1px solid #3b82f6",
              borderRadius: "8px",
              color: "#93c5fd",
              fontSize: "12px",
              cursor: "pointer",
              fontFamily: "inherit",
              letterSpacing: "1px",
              width: "100%",
            }}>
              {showSolution ? "▲ 解答を隠す" : "▼ 解答を表示"}
            </button>

            {showSolution && (
              <Section title="解答（Python）" style={{ marginTop: "12px" }}>
                <CodeBlock code={selected.solution} />
              </Section>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function Section({ title, children, style = {} }) {
  return (
    <div style={{ marginBottom: "20px", ...style }}>
      <div style={{
        fontSize: "10px", color: "#475569", letterSpacing: "2px",
        marginBottom: "8px", textTransform: "uppercase",
      }}>{title}</div>
      {children}
    </div>
  );
}

function CodeBlock({ code }) {
  return (
    <pre style={{
      background: "#0d1117",
      border: "1px solid #1e293b",
      borderRadius: "8px",
      padding: "16px",
      fontSize: "11.5px",
      lineHeight: "1.7",
      color: "#c9d1d9",
      overflowX: "auto",
      margin: 0,
      whiteSpace: "pre",
    }}>{code}</pre>
  );
}
