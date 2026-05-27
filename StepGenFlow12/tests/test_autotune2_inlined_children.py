from src.planner import PlanNode, Tree
from run_autotune2 import _prune_inlined_children_from_tree


def test_prune_inlined_children_removes_subtree(tmp_path):
    grandchild = PlanNode(
        name="grandchild",
        path="root/child/grandchild",
        reference_code="# grandchild",
        refactored_code=None,
        is_leaf=True,
        children=(),
    )
    child = PlanNode(
        name="child",
        path="root/child",
        reference_code="# child",
        refactored_code=None,
        is_leaf=False,
        children=(grandchild,),
    )
    root = PlanNode(
        name="root",
        path="root",
        reference_code="# root",
        refactored_code=None,
        is_leaf=False,
        children=(child,),
    )
    pass1_dir = tmp_path / "pass1" / "iteration_0"
    node_dir = pass1_dir / root.path
    node_dir.mkdir(parents=True)
    (node_dir / "inlined_children.txt").write_text("child\n")

    pruned = _prune_inlined_children_from_tree(Tree(root), pass1_dir)

    assert pruned.root.children == ()
    assert pruned.root.is_leaf
    assert [node.path for node in pruned.iter_topological()] == ["root"]


def test_prune_inlined_children_keeps_live_siblings(tmp_path):
    live = PlanNode(
        name="live",
        path="root/live",
        reference_code="# live",
        refactored_code=None,
        is_leaf=True,
        children=(),
    )
    inlined = PlanNode(
        name="inlined",
        path="root/inlined",
        reference_code="# inlined",
        refactored_code=None,
        is_leaf=True,
        children=(),
    )
    root = PlanNode(
        name="root",
        path="root",
        reference_code="# root",
        refactored_code=None,
        is_leaf=False,
        children=(live, inlined),
    )
    pass1_dir = tmp_path / "pass1" / "iteration_0"
    node_dir = pass1_dir / root.path
    node_dir.mkdir(parents=True)
    (node_dir / "inlined_children.txt").write_text("inlined\n")

    pruned = _prune_inlined_children_from_tree(Tree(root), pass1_dir)

    assert [child.path for child in pruned.root.children] == ["root/live"]
    assert not pruned.root.is_leaf
