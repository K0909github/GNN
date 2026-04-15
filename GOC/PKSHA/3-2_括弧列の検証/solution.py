S = input()
stack = []
pair = {')': '(', ']': '[', '}': '{'}
for c in S:
    if c in '([{':
        stack.append(c)
    else:
        if not stack or stack[-1] != pair[c]:
            print("No")
            exit()
        stack.pop()
print("Yes" if not stack else "No")
