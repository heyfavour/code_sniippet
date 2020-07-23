"""
	用于后台处理list转成tree，
	菜单，目录，树形等
	比较消耗内存，建议对此结构的转换放到js中
"""
data = [
	{'id': 10,'parent_id': 8, 'name': "ACAB"},
	{'id': 9, 'parent_id': 8, 'name': "ACAA"},
	{'id': 8, 'parent_id': 7, 'name': "ACA"},
	{'id': 7, 'parent_id': 1, 'name': "AC"},
	{'id': 6, 'parent_id': 3, 'name': "ABC"},
	{'id': 5, 'parent_id': 3, 'name': "ABB"},
	{'id': 4, 'parent_id': 3, 'name': "ABA"},
	{'id': 3, 'parent_id': 1, 'name': "AB"},
	{'id': 2, 'name': "AA"},
	{'id': 1, 'name': "A"},
]

def list_to_tree(node_list,root_id=None):
	node_dict = {node["id"]:node if node.get("parent_id") else node.update({"parent_id":-1}) or node for node in node_list}
	for node in node_list:node_dict.setdefault(node["parent_id"],{}).setdefault("children",[]).append(node)
	return node_dict[root_id if root_id else -1]

tree = list_to_tree(data)


