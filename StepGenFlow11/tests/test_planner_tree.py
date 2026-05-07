from src.planner import PlanNode, Tree


def _make_simple_tree():
    """Build a small tree by hand:
        root
          child_a (leaf)
          child_b
            grandchild_x (leaf)
    """
    gc = PlanNode(name="grandchild_x", path="root/child_b/grandchild_x",
                  reference_code="<ref_gc>", refactored_code=None,
                  is_leaf=True, children=())
    cb = PlanNode(name="child_b", path="root/child_b",
                  reference_code="<ref_cb>", refactored_code="<ref_cb_refactored>",
                  is_leaf=False, children=(gc,))
    ca = PlanNode(name="child_a", path="root/child_a",
                  reference_code="<ref_ca>", refactored_code=None,
                  is_leaf=True, children=())
    root = PlanNode(name="root", path="root",
                    reference_code="<ref_root>", refactored_code="<ref_root_refactored>",
                    is_leaf=False, children=(ca, cb))
    return Tree(root=root)


def test_iter_leaves_returns_only_leaves():
    tree = _make_simple_tree()
    leaves = list(tree.iter_leaves())
    leaf_paths = sorted(n.path for n in leaves)
    assert leaf_paths == ["root/child_a", "root/child_b/grandchild_x"]


def test_iter_topological_yields_leaves_before_parents():
    tree = _make_simple_tree()
    order = [n.path for n in tree.iter_topological()]
    assert order.index("root/child_a") < order.index("root")
    assert order.index("root/child_b/grandchild_x") < order.index("root/child_b")
    assert order.index("root/child_b") < order.index("root")


def test_find_node_by_path():
    tree = _make_simple_tree()
    node = tree.find("root/child_b/grandchild_x")
    assert node.name == "grandchild_x"
    assert node.is_leaf


def test_find_owner_returns_immediate_parent():
    tree = _make_simple_tree()
    owner = tree.find_owner("root/child_b/grandchild_x")
    assert owner.path == "root/child_b"


def test_find_owner_of_top_child_is_root():
    tree = _make_simple_tree()
    owner = tree.find_owner("root/child_a")
    assert owner.path == "root"


def test_find_owner_of_root_returns_none():
    tree = _make_simple_tree()
    assert tree.find_owner("root") is None
