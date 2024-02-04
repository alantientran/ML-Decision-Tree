def create_decision_tree(data, max_levels):
    my_feature, my_thresh = identify_best_split(data)
    left, right = split_dataset(data, my_feature, my_thresh)

    root_node = TreeNode()
    root_node.feature_idx = my_feature
    root_node.thresh_val = my_thresh
    root_node.is_leaf = False

    root_node.left_child = tree_helper(left, 0, max_levels, root_node)
    root_node.right_child = tree_helper(right, 0, max_levels, root_node)
    
    return root_node

def tree_helper(data_sub, cur_height, max_levels, cur_node) : 
    # Base case - all pure OR we've reached max height
    # todo: what if 'all of the attributes are the same'
    child = TreeNode()
    if (cur_height == max_levels or calc_entropy(data_sub) == 0) : 
        return create_leaf_node(data_sub)
    child.is_leaf = False

    my_feature, my_thresh = identify_best_split(data_sub)

    left, right = split_dataset(data_sub, my_feature, my_thresh)
    child.feature_idx = my_feature
    child.thresh_val = my_thresh

    child.left_child = tree_helper(left, cur_height + 1, max_levels, child)
    child.right_child = tree_helper(right, cur_height + 1, max_levels, child)
    return child


# Test code
test_tree = create_decision_tree(silly_data, 50)
test_tree.printTree()
res = make_prediction(test_tree, silly_data[0])
print(res)


def calc_accuracy(tree_root, data):
    correct = 0
    for dp in range(len(data)) : 
        res = make_prediction(tree_root, data[dp])
        if (res == data[dp].label) :
            correct = correct + 1
     
    return correct / len(data)
tree = create_decision_tree(silly_data, 10)

print(calc_accuracy(tree, silly_data))



def identify_best_split(data):
    # print ("data length in split: ", len(data))
    # print("entropy of the data: ", calc_entropy(data))
    if len(data) < 2:
        print("none time!")
        return (None, None)
    best_feature = 0.0
    best_thresh = 0.0
    best_info_gain = 0.0
    cur_threshold = 0.0 
    cur_info_gain = 0.0
    parent_entropy = calc_entropy(data)
    for ft in range(0, 19) :
        # print("--Check feature ", ft)
        # for dp_idx in range(len(data)) : 
        if data[0].features[ft] > 1 : 
            # This is a continuous feature, so we need to worry about thresholds 
            cur_info_gain, cur_threshold = calc_best_threshold(data, ft)
            # print("cur threshold: ", cur_threshold)
            # print("cur info gain for that threshold: ", cur_info_gain)
        else : 
            # This is a categorical feature. The threshold will always be 0.5 for these features
            cur_threshold = 0.5
            left, right = split_dataset(data, ft, cur_threshold)
            entropy_left = calc_entropy(left)
            entropy_right = calc_entropy(right)
            # print("length of left split: ", len(left))
            # print("length of right split: ", len(right))

            total_entropy = (len(left) / len (data) * entropy_left) + ((len(right) / len(data)) * entropy_right)
            # print("total entropy: ", total_entropy)
            cur_info_gain = parent_entropy - total_entropy
            # print("current info gain: ", cur_info_gain)
        if (cur_info_gain > best_info_gain) :
            best_info_gain = cur_info_gain
            best_feature = ft
            best_thresh = cur_threshold
    return (best_feature, best_thresh)

my_feature, my_thresh = identify_best_split(silly_data)
# print("best feature: ", my_feature, "best thresh: ", my_thresh)


def make_prediction(tree_root, data_point):
    if tree_root.is_leaf : 
        return tree_root.prediction
    if data_point.features[tree_root.feature_idx] >= tree_root.thresh_val : 
        prediction = make_prediction(tree_root.right_child, data_point)
    else : 
         prediction = make_prediction(tree_root.left_child, data_point) 
    return prediction