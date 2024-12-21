class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if len(self.stack) > 0:
            return self.stack.pop()
        else:
            raise IndexError("Empty Stack Pop")

class Queue:
    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.pop(0)
        else:
            raise IndexError("Empty Queue Pop")

class TreeNode:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children if children is not None else {}

class DecisionTree:
    def __init__(self, root):
        self.root = root

    def golf_tree_dfs(self):
        queue = Queue()
        queue.enqueue((self.root, [])) 

        while queue.queue:
            node, path = queue.dequeue()

            if not node.children:
                condition = " and ".join(path)
                print(f"If {condition}, Golf={node.label}")
            else:
                for condition, child in node.children.items():
                    queue.enqueue((child, path + [f"{node.label}={condition}"]))

overcast = TreeNode("yes")
rain_true = TreeNode("no")
rain_false = TreeNode("yes")
sunny_high = TreeNode("no")
sunny_low = TreeNode("yes")

wind = TreeNode("Wind", {"false": rain_false, "true": rain_true})
humidity = TreeNode("Humidity", {"high": sunny_high, "low": sunny_low})
outlook = TreeNode("Outlook", {"rain": wind, "sunny": humidity, "overcast": overcast})

tree = DecisionTree(outlook)

tree.golf_tree_dfs()
