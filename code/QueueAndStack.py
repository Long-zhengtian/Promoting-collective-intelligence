class Stack:
    def __init__(self):
        self.items = []  # 初始化一个列表

    def empty(self):  # 如果为空则返回True 否则返回False
        return self.items == []

    def pop(self):
        try:
            return self.items.pop()
        except Exception:
            raise

    def push(self, item):
        self.items.append(item)

    def top(self):  # 返回栈顶的元素
        return self.items[-1]

    def size(self):
        return len(self.items)

    def get_stack(self):
        return self.items


class Queue:
    def __init__(self):
        self.items = []

    def empty(self):
        return self.items == []

    def push_back(self, item):  # 从队尾插入
        self.items.append(item)

    def pop(self):  # 从队首弹出
        try:
            return self.items.pop(0)
        except Exception:
            raise

    def front(self):
        return self.items[0]

    def size(self):
        return len(self.items)

    def has_item(self, x):
        for i in self.items:
            if x == i:
                return True
        return False

    def get_queue(self):
        return self.items
