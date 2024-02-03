class Node:
  def __init__(self, value, left=None, right=None):
    self.right = right
    self.left = left
    self.value = value

    self.parent = None

    if left:
      self.left.parent = self
    if right:
      self.right.parent = self

  def traverse_preorder(self):
    """
    1. Visit the current node.
    2. Recursively traverse the current node's left subtree.
    3. Recursively traverse the current node's right subtree.
    """
    def traverse(node: Node):
        yield node.value
        if node.left:
            yield from traverse(node.left)
        if node.right:
            yield from traverse(node.right)
    for node in traverse(self):
       yield node

if __name__=="__main__":
    #       a
    #      / \
    #     b   e
    #    / \
    #   c   d
    node = Node('a',
                Node('b',
                     Node('c'),
                     Node('d')),
                Node('e'))
    preorder = ''.join([x for x in node.traverse_preorder()])
    assert 'abcde' == preorder
