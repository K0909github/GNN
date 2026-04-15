import { useState, useEffect, useRef } from "react";

const SETS = [
  {
    set: 1,
    label: "セット 1",
    theme: "基礎力確認",
    problems: [
      {
        id: "1-1", level: "A", levelColor: "#4ade80",
        title: "回文判定",
        timeGuide: "〜10分",
        tags: ["文字列"],
        statement: `文字列 S が与えられます。S を逆順にした文字列と S が一致する場合は "Yes"、そうでない場合は "No" を出力してください。`,
        constraints: [
          "1 ≤ |S| ≤ 10⁵",
          "S は英小文字のみからなる",
        ],
        examples: [
          { input: "racecar", output: "Yes", note: "逆から読んでも同じ" },
          { input: "hello", output: "No", note: "逆にすると 'olleh'" },
          { input: "a", output: "Yes", note: "長さ1は常に回文" },
        ],
        hint: "Python なら s == s[::-1] で一発です。",
        approach: `文字列を逆順にして元の文字列と比較するだけです。\n\n【Python 例】\nS = input()\nprint("Yes" if S == S[::-1] else "No")\n\n計算量: O(|S|)`,
      },
      {
        id: "1-2", level: "B", levelColor: "#facc15",
        title: "予算内で最多購入",
        timeGuide: "15〜20分",
        tags: ["貪欲法", "ソート"],
        statement: `N 個の商品があり、i 番目の商品の値段は p_i 円です。予算 M 円の中でできるだけ多くの商品を購入するとき、最大購入個数を出力してください（各商品は 1 個しか買えません）。`,
        constraints: [
          "1 ≤ N ≤ 2×10⁵",
          "1 ≤ M ≤ 10⁹",
          "1 ≤ p_i ≤ 10⁹",
        ],
        examples: [
          { input: "5 1000\n100 200 300 400 500", output: "4", note: "100+200+300+400=1000" },
          { input: "3 100\n200 300 400", output: "0", note: "最安でも200円で予算オーバー" },
          { input: "4 10\n1 2 3 4", output: "4", note: "1+2+3+4=10 ぴったり" },
        ],
        hint: "安い商品から順番に買っていくとどうなるでしょうか？",
        approach: `値段を昇順ソートし、累積和が M を超えるまで買い続けます。\n\n【Python 例】\nN, M = map(int, input().split())\nP = sorted(map(int, input().split()))\ncount = total = 0\nfor p in P:\n    if total + p <= M:\n        total += p\n        count += 1\n    else:\n        break\nprint(count)\n\n計算量: O(N log N)`,
      },
      {
        id: "1-3", level: "C", levelColor: "#f87171",
        title: "島の個数",
        timeGuide: "30〜40分",
        tags: ["BFS", "DFS", "グラフ"],
        statement: `H 行 W 列のグリッドが与えられます。各マスは '#'（陸地）または '.'（海）のいずれかです。上下左右に隣接する陸地のまとまりを「島」と定義します。グリッド中の島の個数を出力してください。`,
        constraints: [
          "1 ≤ H, W ≤ 1000",
          "各マスは '#' または '.'",
        ],
        examples: [
          { input: "4 5\n##...\n##...\n...##\n...##", output: "2", note: "左上と右下にそれぞれ1つ" },
          { input: "3 3\n#.#\n.#.\n#.#", output: "5", note: "各 '#' は孤立" },
          { input: "2 2\n##\n##", output: "1", note: "全部繋がっている" },
        ],
        hint: "未訪問の '#' を見つけるたびに BFS/DFS で繋がったマスを全て訪問済みにしていきます。",
        approach: `BFS を使う典型問題です。\n\n【Python 例】\nfrom collections import deque\nH, W = map(int, input().split())\ngrid = [input() for _ in range(H)]\nvisited = [[False]*W for _ in range(H)]\ncount = 0\nfor i in range(H):\n    for j in range(W):\n        if grid[i][j] == '#' and not visited[i][j]:\n            count += 1\n            q = deque([(i, j)])\n            visited[i][j] = True\n            while q:\n                r, c = q.popleft()\n                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:\n                    nr, nc = r+dr, c+dc\n                    if 0<=nr<H and 0<=nc<W and grid[nr][nc]=='#' and not visited[nr][nc]:\n                        visited[nr][nc] = True\n                        q.append((nr, nc))\nprint(count)\n\n計算量: O(H×W)`,
      },
    ],
  },
  {
    set: 2,
    label: "セット 2",
    theme: "アルゴリズム応用",
    problems: [
      {
        id: "2-1", level: "A", levelColor: "#4ade80",
        title: "最大部分列和",
        timeGuide: "15〜20分",
        tags: ["動的計画法", "Kadane"],
        statement: `N 個の整数からなる数列 a₁, a₂, ..., aₙ が与えられます。空でない連続部分列の和の最大値を出力してください。`,
        constraints: [
          "1 ≤ N ≤ 2×10⁵",
          "−10⁹ ≤ aᵢ ≤ 10⁹",
        ],
        examples: [
          { input: "9\n-2 1 -3 4 -1 2 1 -5 4", output: "6", note: "[4, -1, 2, 1] の和が最大" },
          { input: "1\n-5", output: "-5", note: "要素が1つなので選ぶしかない" },
          { input: "4\n1 2 3 4", output: "10", note: "全要素の和" },
        ],
        hint: "Kadane のアルゴリズム：現在の連続和が負になったらリセットするとどうなりますか？",
        approach: `Kadane's Algorithm の典型です。\n\n【Python 例】\nN = int(input())\nA = list(map(int, input().split()))\ncur = best = A[0]\nfor a in A[1:]:\n    cur = max(a, cur + a)\n    best = max(best, cur)\nprint(best)\n\n直感: cur が負なら捨てて次の要素から再スタート\n計算量: O(N)`,
      },
      {
        id: "2-2", level: "B", levelColor: "#facc15",
        title: "区間スケジューリング",
        timeGuide: "25〜35分",
        tags: ["貪欲法", "区間"],
        statement: `N 個のタスクがあり、i 番目のタスクは時刻 sᵢ に開始し時刻 eᵢ に終了します（同じ時刻に終了・開始するタスクは重複なく連続して実行できます）。1 人の作業者が重複なく実行できるタスクの最大個数を出力してください。`,
        constraints: [
          "1 ≤ N ≤ 2×10⁵",
          "0 ≤ sᵢ < eᵢ ≤ 10⁹",
        ],
        examples: [
          { input: "4\n1 3\n2 5\n3 9\n6 8", output: "3", note: "[1,3], [3,9] は重複なので [1,3],[6,8] or [1,3],[3,9] → 3個" },
          { input: "3\n1 10\n2 3\n4 5", output: "2", note: "[2,3] と [4,5] の2つ" },
          { input: "1\n0 1", output: "1" },
        ],
        hint: "終了時刻が早いタスクから貪欲に選ぶのが最適解です。",
        approach: `終了時刻でソートして貪欲に選択します。\n\n【Python 例】\nN = int(input())\ntasks = [tuple(map(int, input().split())) for _ in range(N)]\ntasks.sort(key=lambda x: x[1])  # 終了時刻でソート\ncount = 0\nlast_end = -1\nfor s, e in tasks:\n    if s >= last_end:\n        count += 1\n        last_end = e\nprint(count)\n\n計算量: O(N log N)\n※ 証明: 終了が早いタスクを選ぶと「残り時間」が最大化される`,
      },
      {
        id: "2-3", level: "C", levelColor: "#f87171",
        title: "コスト付き迷路の最短路",
        timeGuide: "40〜50分",
        tags: ["ダイクストラ法", "グラフ"],
        statement: `H 行 W 列のグリッドがあり、マス (i, j) に入るコストは aᵢⱼ です。左上 (0, 0) から右下 (H-1, W-1) まで上下左右に移動するとき、通過するマスの入室コスト合計の最小値を出力してください（スタート地点のコストも含む）。`,
        constraints: [
          "1 ≤ H, W ≤ 1000",
          "1 ≤ aᵢⱼ ≤ 10⁹",
        ],
        examples: [
          { input: "3 3\n1 3 1\n1 5 1\n4 2 1", output: "7", note: "(0,0)→(0,2)→(1,2)→(2,2): 1+1+1+1+1+1+1=7" },
          { input: "1 1\n5", output: "5", note: "スタートゴールが同じ" },
          { input: "2 2\n1 2\n3 4", output: "7", note: "1→2→4=7、1→3→4=8 より最小は7" },
        ],
        hint: "dist[i][j] を (0,0) からマス (i,j) までの最小コストとして管理します。優先度付きキューが必要です。",
        approach: `ダイクストラ法の適用です。\n\n【Python 例】\nimport heapq\nH, W = map(int, input().split())\nA = [list(map(int, input().split())) for _ in range(H)]\nINF = float('inf')\ndist = [[INF]*W for _ in range(H)]\ndist[0][0] = A[0][0]\nhq = [(A[0][0], 0, 0)]\nwhile hq:\n    d, r, c = heapq.heappop(hq)\n    if d > dist[r][c]:\n        continue\n    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:\n        nr, nc = r+dr, c+dc\n        if 0<=nr<H and 0<=nc<W:\n            nd = d + A[nr][nc]\n            if nd < dist[nr][nc]:\n                dist[nr][nc] = nd\n                heapq.heappush(hq, (nd, nr, nc))\nprint(dist[H-1][W-1])\n\n計算量: O(HW log HW)`,
      },
    ],
  },
  {
    set: 3,
    label: "セット 3",
    theme: "数学・DP",
    problems: [
      {
        id: "3-1", level: "A", levelColor: "#4ade80",
        title: "約数の個数",
        timeGuide: "10〜15分",
        tags: ["数学", "整数論"],
        statement: `整数 N が与えられます。N の約数の個数を出力してください。`,
        constraints: [
          "1 ≤ N ≤ 10¹²",
        ],
        examples: [
          { input: "12", output: "6", note: "約数は 1,2,3,4,6,12" },
          { input: "1", output: "1", note: "約数は 1 のみ" },
          { input: "1000000007", output: "2", note: "素数なので約数は 1 と自身のみ" },
        ],
        hint: "N が最大 10¹² なので全探索は無理です。√N まで試し割りして、i が約数なら N/i も約数であることを使います。",
        approach: `i=1 から √N まで試し割りします。\n\n【Python 例】\nimport math\nN = int(input())\ncount = 0\ni = 1\nwhile i * i <= N:\n    if N % i == 0:\n        count += 2 if i * i != N else 1\n    i += 1\nprint(count)\n\n計算量: O(√N) ≈ O(10⁶) で間に合います`,
      },
      {
        id: "3-2", level: "B", levelColor: "#facc15",
        title: "括弧列の検証",
        timeGuide: "20〜25分",
        tags: ["スタック", "文字列"],
        statement: `'('、')'、'['、']'、'{'、'}' のみからなる文字列 S が与えられます。S が正しい括弧の対応になっているかを判定し、"Yes" または "No" を出力してください。正しい括弧列とは、全ての開き括弧に対応する閉じ括弧が正しい順序で存在するものです。`,
        constraints: [
          "1 ≤ |S| ≤ 10⁵",
          "S は '('、')'、'['、']'、'{'、'}' のみ",
        ],
        examples: [
          { input: "()[]{}", output: "Yes" },
          { input: "([{}])", output: "Yes" },
          { input: "(]", output: "No", note: "対応が違う" },
          { input: "([)]", output: "No", note: "入れ子が崩れている" },
          { input: "{[]}", output: "Yes" },
        ],
        hint: "スタックを使いましょう。開き括弧はスタックに積み、閉じ括弧はスタックトップと照合します。",
        approach: `スタックを使う典型問題です。\n\n【Python 例】\nS = input()\nstack = []\npair = {')': '(', ']': '[', '}': '{'}\nfor c in S:\n    if c in '([{':\n        stack.append(c)\n    else:\n        if not stack or stack[-1] != pair[c]:\n            print("No")\n            exit()\n        stack.pop()\nprint("Yes" if not stack else "No")\n\n計算量: O(|S|)`,
      },
      {
        id: "3-3", level: "C", levelColor: "#f87171",
        title: "コイン最小枚数（無制限）",
        timeGuide: "30〜40分",
        tags: ["動的計画法", "コイン問題"],
        statement: `N 種類のコインがあり、i 番目のコインの価値は cᵢ 円です。金額 M 円をちょうど支払うために必要なコインの最小枚数を出力してください。支払いが不可能な場合は -1 を出力してください（各コインは無制限に使えます）。`,
        constraints: [
          "1 ≤ N ≤ 10³",
          "1 ≤ M ≤ 10⁴",
          "1 ≤ cᵢ ≤ 10⁴",
        ],
        examples: [
          { input: "3 11\n1 5 6", output: "2", note: "5+6=11 で 2 枚" },
          { input: "2 3\n2 4", output: "-1", note: "2 と 4 では 3 を作れない" },
          { input: "1 10000\n1", output: "10000", note: "1 円玉 10000 枚" },
        ],
        hint: "dp[j] = 金額 j を作る最小コイン枚数 と定義します。dp[0] = 0 から始めてみましょう。",
        approach: `完全ナップサック問題（無制限コイン）のDPです。\n\n【Python 例】\nN, M = map(int, input().split())\nC = list(map(int, input().split()))\nINF = float('inf')\ndp = [INF] * (M + 1)\ndp[0] = 0\nfor j in range(1, M + 1):\n    for c in C:\n        if j >= c and dp[j - c] != INF:\n            dp[j] = min(dp[j], dp[j - c] + 1)\nprint(dp[M] if dp[M] != INF else -1)\n\n計算量: O(N×M)\n※ 遷移: 金額 j を作るには「j-c を作って c を1枚足す」`,
      },
    ],
  },
];

const LEVEL_STYLE = {
  A: { bg: "bg-emerald-500/20", text: "text-emerald-400", border: "border-emerald-500/40" },
  B: { bg: "bg-yellow-500/20", text: "text-yellow-400", border: "border-yellow-500/40" },
  C: { bg: "bg-red-500/20", text: "text-red-400", border: "border-red-500/40" },
};

function formatTime(s) {
  const m = Math.floor(s / 60).toString().padStart(2, "0");
  const sec = (s % 60).toString().padStart(2, "0");
  return `${m}:${sec}`;
}

export default function App() {
  const [activeSet, setActiveSet] = useState(0);
  const [activeProblem, setActiveProblem] = useState(0);
  const [showHint, setShowHint] = useState({});
  const [showApproach, setShowApproach] = useState({});
  const [timerActive, setTimerActive] = useState(false);
  const [timeLeft, setTimeLeft] = useState(150 * 60);
  const intervalRef = useRef(null);

  const currentSet = SETS[activeSet];
  const problem = currentSet.problems[activeProblem];
  const ls = LEVEL_STYLE[problem.level];

  useEffect(() => {
    if (timerActive && timeLeft > 0) {
      intervalRef.current = setInterval(() => setTimeLeft(t => t - 1), 1000);
    } else if (!timerActive) {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [timerActive, timeLeft]);

  const resetTimer = () => {
    setTimerActive(false);
    setTimeLeft(150 * 60);
  };

  const toggleHint = (id) => setShowHint(p => ({ ...p, [id]: !p[id] }));
  const toggleApproach = (id) => setShowApproach(p => ({ ...p, [id]: !p[id] }));

  const timerColor = timeLeft < 600 ? "text-red-400" : timeLeft < 1800 ? "text-yellow-400" : "text-emerald-400";

  return (
    <div style={{ fontFamily: "'Courier New', monospace", background: "#080c14", minHeight: "100vh", color: "#c9d1d9" }}>
      {/* Header */}
      <div style={{ background: "linear-gradient(90deg, #0d1117 0%, #111827 100%)", borderBottom: "1px solid #1f2937", padding: "16px 24px", display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ background: "linear-gradient(135deg, #4ade80, #22d3ee)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", fontWeight: 700, fontSize: 20, letterSpacing: 2 }}>PKSHA</span>
            <span style={{ color: "#6b7280", fontSize: 14 }}>Software Engineer</span>
            <span style={{ background: "#1f2937", color: "#9ca3af", fontSize: 11, padding: "2px 8px", borderRadius: 4, border: "1px solid #374151" }}>コーディングテスト対策</span>
          </div>
          <div style={{ color: "#4b5563", fontSize: 12, marginTop: 4 }}>3問 / 150分 想定 ｜ AtCoder A〜C レベル相当</div>
        </div>
        {/* Timer */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, background: "#0d1117", border: "1px solid #1f2937", borderRadius: 8, padding: "8px 16px" }}>
          <div>
            <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 2 }}>本番タイマー</div>
            <div className={timerColor} style={{ fontSize: 28, fontWeight: 700, letterSpacing: 2, color: timeLeft < 600 ? "#f87171" : timeLeft < 1800 ? "#facc15" : "#4ade80" }}>{formatTime(timeLeft)}</div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <button onClick={() => setTimerActive(t => !t)} style={{ background: timerActive ? "#1f2937" : "#14532d", color: timerActive ? "#9ca3af" : "#4ade80", border: "1px solid", borderColor: timerActive ? "#374151" : "#16a34a", borderRadius: 6, padding: "4px 12px", cursor: "pointer", fontSize: 12, fontFamily: "'Courier New', monospace" }}>
              {timerActive ? "⏸ 停止" : "▶ 開始"}
            </button>
            <button onClick={resetTimer} style={{ background: "transparent", color: "#6b7280", border: "1px solid #374151", borderRadius: 6, padding: "4px 12px", cursor: "pointer", fontSize: 12, fontFamily: "'Courier New', monospace" }}>↺ リセット</button>
          </div>
        </div>
      </div>

      <div style={{ display: "flex", maxWidth: 1100, margin: "0 auto", padding: 24, gap: 20, flexWrap: "wrap" }}>
        {/* Left sidebar */}
        <div style={{ width: 220, flexShrink: 0 }}>
          {/* Set selector */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ color: "#6b7280", fontSize: 11, letterSpacing: 1, marginBottom: 8, textTransform: "uppercase" }}>練習セット</div>
            {SETS.map((s, si) => (
              <button key={si} onClick={() => { setActiveSet(si); setActiveProblem(0); }}
                style={{ width: "100%", textAlign: "left", background: activeSet === si ? "#111827" : "transparent", border: activeSet === si ? "1px solid #1f2937" : "1px solid transparent", borderRadius: 8, padding: "10px 14px", cursor: "pointer", marginBottom: 6, transition: "all 0.15s", color: activeSet === si ? "#e5e7eb" : "#6b7280", fontFamily: "'Courier New', monospace" }}>
                <div style={{ fontSize: 13, fontWeight: activeSet === si ? 700 : 400 }}>{s.label}</div>
                <div style={{ fontSize: 11, color: "#4b5563", marginTop: 2 }}>{s.theme}</div>
              </button>
            ))}
          </div>
          {/* Problem selector */}
          <div>
            <div style={{ color: "#6b7280", fontSize: 11, letterSpacing: 1, marginBottom: 8, textTransform: "uppercase" }}>問題</div>
            {currentSet.problems.map((p, pi) => {
              const l = LEVEL_STYLE[p.level];
              return (
                <button key={pi} onClick={() => setActiveProblem(pi)}
                  style={{ width: "100%", textAlign: "left", background: activeProblem === pi ? "#111827" : "transparent", border: activeProblem === pi ? "1px solid #1f2937" : "1px solid transparent", borderRadius: 8, padding: "10px 14px", cursor: "pointer", marginBottom: 6, transition: "all 0.15s", fontFamily: "'Courier New', monospace" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 6px", borderRadius: 4, border: "1px solid", background: activeProblem === pi ? (p.level === "A" ? "#052e16" : p.level === "B" ? "#422006" : "#450a0a") : "#1f2937", color: p.level === "A" ? "#4ade80" : p.level === "B" ? "#facc15" : "#f87171", borderColor: p.level === "A" ? "#16a34a" : p.level === "B" ? "#ca8a04" : "#dc2626" }}>{p.level}</span>
                    <span style={{ fontSize: 12, color: activeProblem === pi ? "#e5e7eb" : "#6b7280" }}>問題 {pi + 1}</span>
                  </div>
                  <div style={{ fontSize: 12, color: activeProblem === pi ? "#9ca3af" : "#4b5563", marginTop: 4, fontWeight: activeProblem === pi ? 600 : 400 }}>{p.title}</div>
                </button>
              );
            })}
          </div>

          {/* Progress info */}
          <div style={{ marginTop: 20, background: "#0d1117", border: "1px solid #1f2937", borderRadius: 8, padding: 12 }}>
            <div style={{ color: "#6b7280", fontSize: 11, marginBottom: 8 }}>目安時間</div>
            {currentSet.problems.map((p, pi) => (
              <div key={pi} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4, color: activeProblem === pi ? "#e5e7eb" : "#4b5563" }}>
                <span>問{pi+1} ({p.level})</span>
                <span>{p.timeGuide}</span>
              </div>
            ))}
            <div style={{ borderTop: "1px solid #1f2937", marginTop: 8, paddingTop: 8, display: "flex", justifyContent: "space-between", fontSize: 11, color: "#9ca3af" }}>
              <span>合計</span>
              <span>〜90分</span>
            </div>
          </div>
        </div>

        {/* Main content */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {/* Problem header */}
          <div style={{ background: "#0d1117", border: "1px solid #1f2937", borderRadius: 12, padding: "20px 24px", marginBottom: 16 }}>
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: 12, marginBottom: 12 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <span style={{ fontSize: 11, fontWeight: 700, padding: "4px 10px", borderRadius: 6, border: "1px solid", background: problem.level === "A" ? "#052e16" : problem.level === "B" ? "#422006" : "#450a0a", color: problem.level === "A" ? "#4ade80" : problem.level === "B" ? "#facc15" : "#f87171", borderColor: problem.level === "A" ? "#16a34a" : problem.level === "B" ? "#ca8a04" : "#dc2626" }}>
                  {problem.level === "A" ? "A レベル" : problem.level === "B" ? "B レベル" : "C レベル"}
                </span>
                <h1 style={{ fontSize: 20, fontWeight: 700, color: "#f0f6fc", margin: 0 }}>問題 {activeProblem + 1}：{problem.title}</h1>
              </div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {problem.tags.map(t => (
                  <span key={t} style={{ background: "#1f2937", color: "#9ca3af", fontSize: 11, padding: "3px 8px", borderRadius: 4, border: "1px solid #374151" }}>{t}</span>
                ))}
              </div>
            </div>
            <div style={{ color: "#6b7280", fontSize: 12 }}>⏱ 目安: {problem.timeGuide}</div>
          </div>

          {/* Problem statement */}
          <div style={{ background: "#0d1117", border: "1px solid #1f2937", borderRadius: 12, padding: "20px 24px", marginBottom: 16 }}>
            <div style={{ color: "#6b7280", fontSize: 11, letterSpacing: 1, marginBottom: 10, textTransform: "uppercase" }}>問題文</div>
            <p style={{ color: "#e5e7eb", lineHeight: 1.8, fontSize: 14, margin: 0 }}>{problem.statement}</p>
          </div>

          {/* Constraints */}
          <div style={{ background: "#0d1117", border: "1px solid #1f2937", borderRadius: 12, padding: "20px 24px", marginBottom: 16 }}>
            <div style={{ color: "#6b7280", fontSize: 11, letterSpacing: 1, marginBottom: 10, textTransform: "uppercase" }}>制約</div>
            {problem.constraints.map((c, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <span style={{ color: "#4ade80", fontSize: 10 }}>▸</span>
                <span style={{ fontSize: 13, color: "#d1d5db", fontFamily: "'Courier New', monospace" }}>{c}</span>
              </div>
            ))}
          </div>

          {/* Examples */}
          <div style={{ background: "#0d1117", border: "1px solid #1f2937", borderRadius: 12, padding: "20px 24px", marginBottom: 16 }}>
            <div style={{ color: "#6b7280", fontSize: 11, letterSpacing: 1, marginBottom: 14, textTransform: "uppercase" }}>入出力例</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {problem.examples.map((ex, i) => (
                <div key={i} style={{ background: "#080c14", border: "1px solid #1f2937", borderRadius: 8, overflow: "hidden" }}>
                  <div style={{ display: "flex", gap: 0, flexWrap: "wrap" }}>
                    <div style={{ flex: 1, padding: "12px 16px", borderRight: "1px solid #1f2937", minWidth: 140 }}>
                      <div style={{ color: "#6b7280", fontSize: 10, marginBottom: 6 }}>入力 {i + 1}</div>
                      <pre style={{ color: "#93c5fd", fontSize: 12, margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-all" }}>{ex.input}</pre>
                    </div>
                    <div style={{ flex: 1, padding: "12px 16px", minWidth: 80 }}>
                      <div style={{ color: "#6b7280", fontSize: 10, marginBottom: 6 }}>出力 {i + 1}</div>
                      <pre style={{ color: "#4ade80", fontSize: 12, margin: 0 }}>{ex.output}</pre>
                    </div>
                  </div>
                  {ex.note && (
                    <div style={{ padding: "8px 16px", borderTop: "1px solid #1f2937", color: "#6b7280", fontSize: 11 }}>
                      💬 {ex.note}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Hint */}
          <div style={{ marginBottom: 12 }}>
            <button onClick={() => toggleHint(problem.id)} style={{ width: "100%", textAlign: "left", background: showHint[problem.id] ? "#111827" : "#0d1117", border: "1px solid", borderColor: showHint[problem.id] ? "#ca8a04" : "#1f2937", borderRadius: 12, padding: "14px 20px", cursor: "pointer", color: "#facc15", fontSize: 13, fontFamily: "'Courier New', monospace", transition: "all 0.2s", display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 16 }}>{showHint[problem.id] ? "▾" : "▸"}</span>
              <span style={{ fontWeight: 600 }}>💡 ヒント</span>
              <span style={{ color: "#6b7280", fontSize: 11, marginLeft: "auto" }}>{showHint[problem.id] ? "閉じる" : "表示する"}</span>
            </button>
            {showHint[problem.id] && (
              <div style={{ background: "#0d1117", border: "1px solid #ca8a04", borderTop: "none", borderRadius: "0 0 12px 12px", padding: "16px 20px" }}>
                <p style={{ color: "#fde68a", fontSize: 13, margin: 0, lineHeight: 1.7 }}>{problem.hint}</p>
              </div>
            )}
          </div>

          {/* Approach */}
          <div style={{ marginBottom: 12 }}>
            <button onClick={() => toggleApproach(problem.id)} style={{ width: "100%", textAlign: "left", background: showApproach[problem.id] ? "#111827" : "#0d1117", border: "1px solid", borderColor: showApproach[problem.id] ? "#7c3aed" : "#1f2937", borderRadius: 12, padding: "14px 20px", cursor: "pointer", color: "#a78bfa", fontSize: 13, fontFamily: "'Courier New', monospace", transition: "all 0.2s", display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 16 }}>{showApproach[problem.id] ? "▾" : "▸"}</span>
              <span style={{ fontWeight: 600 }}>🔍 解法・サンプルコード</span>
              <span style={{ color: "#6b7280", fontSize: 11, marginLeft: "auto" }}>{showApproach[problem.id] ? "閉じる" : "表示する（ネタバレ注意）"}</span>
            </button>
            {showApproach[problem.id] && (
              <div style={{ background: "#0d1117", border: "1px solid #7c3aed", borderTop: "none", borderRadius: "0 0 12px 12px", padding: "16px 20px" }}>
                <pre style={{ color: "#ddd6fe", fontSize: 12, margin: 0, whiteSpace: "pre-wrap", lineHeight: 1.8 }}>{problem.approach}</pre>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div style={{ borderTop: "1px solid #1f2937", padding: "12px 24px", textAlign: "center", color: "#374151", fontSize: 11 }}>
        PKSHA Technology ソフトウェアエンジニア コーディングテスト対策 ｜ AtCoder A〜C 相当 ｜ 3問 150分
      </div>
    </div>
  );
}
