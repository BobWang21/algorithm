#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:39:49 2019

@author: wangbao
"""

# Definition for a binary tree node.
class TreeNode(object):
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None

def construct_from_list(lst):
    if not lst:
        return 
    root = TreeNode(lst[0])
    queue = [root]
    i = 1
    while i < len(lst):
        if not queue[0].left:
          new_tree = TreeNode(lst[i])  
          queue[0].left = new_tree 
          queue.append(new_tree)
          i += 1
          continue
        if not queue[0].right: 
          new_tree = TreeNode(lst[i])  
          queue[0].right = new_tree 
          queue.append(new_tree)
          queue.pop(0) # 移除该节点
          i += 1
    return root


# 先序遍历 递归
def preorder_traversal(root):
    if root:
       print(root.val)
       preorder_traversal(root.left) 
       preorder_traversal(root.right)       
       
# 先序遍历 非递归->栈
def preorderTraversal(root):
    ret = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node:
            ret.append(node.val)
            stack.append(node.right) # 先放右子树
            stack.append(node.left)
    
    return ret 

# 中序遍历 非递归 all the way down  
def inorderTraversal(root): 
    if not root:
        return 
    node = root 
    stack = []
    ret = []
    while node or stack:
        while node: # 一路向左
            stack.append(node)
            node = node.left
        tail = stack.pop()
        ret.append(tail.val)
        node = tail.right
    return ret 

# 层次遍历1 使用队列, 先进先出
def level_traversal(root):
    if not root:
        return
    queue = [root]
    ret = []
    while queue:
        head = queue.pop(0)
        ret.append(head.val)
        l, r = head.left, head.right
        if l: queue.append(l)     
        if r: queue.append(r)
    return ret        

# 层次遍历2
def level_traversal_2(root):
    if not root:
        return []
    curr_level = [root]
    ret = []
    while curr_level:
        next_level = []
        for tree in curr_level: #访问当前层的每个节点, 并将叶节点放入下一层
            ret.append(tree.val)
            left = tree.left
            right = tree.right
            if left:
                next_level.append(left)
            if right:
                next_level.append(right)    
        curr_level = next_level
    return ret              
        
# 层数
def level_num(root):
    if not root:
        return 0
    return 1 + max(level_num(root.left), level_num(root.right))

# 公共祖先
def lowestCommonAncestor(root, p, q):
     # base 结束条件
     if not root or root.val == p or root.val == q: 
         return root
     left = lowestCommonAncestor(root.left, p, q)
     right = lowestCommonAncestor(root.right, p, q)
     if left and right: # 在不同的分支中
         return root
     return left if left else right


# BST
def lowestCommonAncestor2(root, p, q):
    if not root:
        return
    if root.val == p or root.val == q:
        return root.val
    if root.val > max(p, q):
        return lowestCommonAncestor(root.left, p, q)
    if root.val < min(p, q):
        return lowestCommonAncestor(root.right, p, q)
    return root


# preorder = [3,9,20,15,7]
# inorder = [9,3,15,20,7]    
def buildTree(preorder, inorder):
    if not preorder or not inorder:
        return None
    root_value = preorder[0]
    root_index = inorder.index(root_value)
    
    left_inorder = inorder[:root_index]
    right_inorder = inorder[root_index+1: ]
    
    left_preorder = preorder[1: len(left_inorder)+1]
    right_preorder = preorder[len(left_inorder)+1:]
    print(root_value, left_inorder, right_inorder, left_preorder, right_preorder)
    root = TreeNode(root_value)
    root.left = buildTree(left_preorder, left_inorder)
    root.r = buildTree(right_preorder, right_inorder)
    return root

## 判断一个树是否平衡
def balanceTree(root):
    if abs(level_num(root.left) -  level_num(root.right)) > 1:
        return False
    return True

## 典型 返回两种信息 1 子树高度 2 子树是否平衡
def balanceTree_(root):
    def check(root):
        if not root:
           return (True, 0)
        l = check(root.left)
        r = check(root.right)
        if not l[0] or not r[0]:
            return (False, max(l[1],  r[1])+1)
        if abs(l[1] - r[1]) <= 1:
            return (True, max(l[1],  r[1]) + 1)
        return (False, max(l[1],  r[1]) + 1)
    return check(root)[0]



class Solution(object):
    def __init__(self):
        self.total = 0

    def convertBST(self, root):
        if root is not None:
            self.convertBST(root.right)
            self.total += root.val
            root.val = self.total
            self.convertBST(root.left)
        return root

def hasPathSum(root, sum):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: bool
    """
    if not root:
        return []
    left, right = root.left, root.root.left
    if not left and not right and root.val == sum:
        return [[root.val]]
    path = hasPathSum(left, sum - root.val) + hasPathSum(right, sum - root.val)
    return [[root.val] + p for p in path]



def rightSideView(root):
    if not root:
        return []
    curr_level = [root]
    ret = []
    while curr_level:
        next_level = []
        for tree in curr_level:
            left = tree.left
            right = tree.right
            if left:
                next_level.append(left)
            if right:
                next_level.append(right)    
        ret.append(tree.val) 
        curr_level = next_level
    return ret 

    



class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        if not root:
            return []
        paths = self.binaryTreePaths(root.left) + self.binaryTreePaths(root.right)
        return [str(root.val) + '->' + path for path in paths]

# 叶子节点的判断!!!            
def binaryTreePaths(root):
    """
    :type root: TreeNode
    :rtype: List[str]
    """
    if not root:
        return []
    if not root.left and not root.right: # xxx
        return [str(root.val)]
    paths = binaryTreePaths(root.left) + binaryTreePaths(root.right)
    return [str(root.val) + '->' + path for path in paths]

def findFrequentTreeSum(root):
    """
    :type root: TreeNod
    :rtype: List[int]
    """
    def frequent(root):
        if not root:
            return []
        if not root.left and not root.right:
            return [root.val]
        if not root.left:
            return [root.val + frequent(root.right)[0]] + frequent(root.right)
        if not root.right:
            return [root.val + frequent(root.left)[0]] + frequent(root.left)
        return [root.val + frequent(root.left)[0] + frequent(root.right)[0]] + frequent(root.left) + frequent(root.right)
    frequent_list = frequent(root)
    
    if not frequent_list:
        return frequent_list
    
    dic = {}
    for val in frequent_list:
        dic.setdefault(val, 0)
        dic[val] += 1
    
    max_num = 0    
    for key in dic:
        max_num = max(max_num, dic[key])
    
    ret = []    
    for key in dic:
        if dic[key] == max_num:
            ret.append(key)
    return ret        
        
class Solution(object):
    def findFrequentTreeSum(self, root):
        """
        :type root: TreeNode
        :rtype: List[int] 使用字典计数
        """
        def helper(root, d): 
            if not root:
                return 0
            left = helper(root.left, d)
            right = helper(root.right, d)
            subtreeSum = left + right + root.val
            d[subtreeSum] = d.get(subtreeSum, 0) + 1
            return subtreeSum
        
        d = {}
        helper(root, d)
        mostFreq = 0
        ans = []
        for key in d:
            if d[key] > mostFreq:
                mostFreq = d[key]
                ans = [key]
            elif d[key] == mostFreq:
                ans.append(key)
        return ans



class Solution(object):
    def largestValues(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        ret = []
        if not root:
            return ret
        now_level = [root]
        while now_level:
            next_level = [] 
            max_num = now_level[0].val
            for node in now_level:
                max_num = max(max_num, node.val)
                if node.left:
                    next_level.append(node.left)
                if node.right:   
                   next_level.append(node.right)
            now_level = next_level
            ret.append(max_num)
        
        return ret  
    
        
class Solution(object):
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        ret = []
        if not root:
            return ret
        now_level = [root]
        while now_level:
            next_level = [] 
            max_num = now_level[0].val
            for node in now_level:
                max_num = max(max_num, node.val)
                if node.left:
                    next_level.append(node.left)
                if node.right:   
                   next_level.append(node.right)
            now_level = next_level
            ret.append(max_num)
        
        return ret 
    
    
###  convertBST  
class Solution(object):
    def convertBST(self, root):
        total = 0
        
        node = root
        stack = []
        while stack or node is not None:
            # push all nodes up to (and including) this subtree's maximum on
            # the stack.
            while node is not None:
                stack.append(node)
                node = node.right

            node = stack.pop()
            total += node.val
            node.val = total

            # all nodes with values between the current and its parent lie in
            # the left subtree.
            node = node.left

        return root
    
    
class Solution(object):
    def convertBST(self, root):
        total = 0
        
        node = root
        stack = []
        while stack or node is not None:
            # push all nodes up to (and including) this subtree's maximum on
            # the stack.
            while node is not None:
                stack.append(node)
                node = node.right

            node = stack.pop()
            total += node.val
            node.val = total

            # all nodes with values between the current and its parent lie in
            # the left subtree.
            node = node.left

        return root    


# 一棵树的最长路径 边数
class Solution(object):
    def diameterOfBinaryTree(self, root):
        self.ans = 1
        def depth(node):
            if not node: return 0
            L = depth(node.left)
            R = depth(node.right)
            self.ans = max(self.ans, L+R+1)
            return max(L, R) + 1

        depth(root)
        return self.ans - 1 


class Solution(object):
    def maxDepth(self, root):
        if root == None:
            return 0
        depth = 0
        stack = [root]
        while stack:
            next_level = []
            while stack:
                node = stack.pop()
                if node.children:
                    next_level += node.children
            stack = next_level
            depth += 1
        return depth

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: Node
        :rtype: int
        """
        if not root:
            return 0
        now_level = [root]
        i = 0
        while now_level:
            next_level = []
            for node in now_level:
                if node.children:
                    next_level+= node.children
            i += 1
            next_level = now_level
        return i    



class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        now_level = [root]
        i = 0
        while now_level:
            next_level = []
            i += 1
            for node in now_level:
                if not node.left and not node.right:
                    return i
                next_level.append(node.left)
                next_level.append(node.right)
            now_level = next_level 
        return i             
        
        
    

    
        
    
        
          
        