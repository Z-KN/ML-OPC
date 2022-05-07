# This is the file dealing with commands
with open(file='recipe.txt', mode='r') as file:
    recipe = file.readlines()

seg_rules_dict = {}  # use a dictionary to store rules
seg_cmd_list = []  # use a list to store segmentation
for command in recipe:
    option = command.split()
    if not option:  # empty?
        continue
    # print(option)
    command_type = option[0]
    # Process different types of commands
    # assert command_type in ['Seg_Edge', 'Seg_Rule']  
    if command_type == 'Seg_Rule':
        rule_dict = {}
        rule_name = '_default_'
        if '-name' in option:
            name_index = option.index('-name')
            rule_name = option[name_index+1]
            # The name of this rule is special
        if '-type' in option:
            type_index = option.index('-type')
            rule_type = option[type_index+1]
            rule_dict['type'] = rule_type
        if '-segnum' in option:
            segnum_index = option.index('-segnum')
            rule_segnum = int(option[segnum_index+1])
            rule_dict['segnum'] = rule_segnum
        if '-segpara' in option:
            segpara_index = option.index('-segpara')
            rule_segpara = option[segpara_index+1:segpara_index+3*rule_segnum+1]
            rule_dict['segpara'] = rule_segpara
        if '-eslen' in option:
            eslen_index = option.index('-eslen')
            rule_eslen = int(option[eslen_index+1])
            rule_dict['eslen'] = rule_eslen
        if '-mark' in option:
            mark_index = option.index('-mark')
            rule_mark = option[mark_index+1]
            rule_dict['mark'] = rule_mark
        seg_rules_dict[rule_name] = rule_dict  # nested dict
    if command_type == 'Default_Outer_Rule':
        default_outer_rule = option[1]
    if command_type == 'Default_Inner_Rule':
        default_inner_rule = option[1]
    if command_type == 'Default_Direct_Rule':
        default_direct_rule = option[1]
    if command_type == 'Default_Middel_Rule':
        default_middle_rule = option[1]
    if command_type == 'Default_Conflict_Rule':
        default_conflict_rule = option[1]
    if command_type == 'Seg_Edge':
        seg = {}
        direction = option[1]
        in_end = option[2]
        out_end = option[3]
        seg['direction'] = direction
        seg['in_end'] = in_end
        seg['out_end'] = out_end
        assert direction == 'XY' or 'ANY'  # must be horizontal/vertical
        # Deal with length
        if '-len' in option:
            len_index = option.index('-len')
            range_low = float(option[len_index+1])
            range_high = float(option[len_index+2])
            if range_high == 'INF':  # convert to number
                range_high == float('+inf')
            seg['len'] = (range_low, range_high)  # the value is a tuple
        if '-iseg' in option:
            iseg_index = option.index('-iseg')
            iseg_rule = option[iseg_index+1]
            seg['iseg'] = iseg_rule
        if '-oseg' in option:
            oseg_index = option.index('-oseg')
            oseg_rule = option[oseg_index+1]
            seg['oseg'] = oseg_rule
        if '-mseg' in option:
            mseg_index = option.index('-mseg')
            mseg_rule = option[mseg_index+1]
            seg['mseg'] = mseg_rule
        if '-wseg' in option:
            wseg_index = option.index('-wseg')
            wseg_rule = option[wseg_index+1]
            seg['wseg'] = wseg_rule
        if '-idefault' in option:
            iseg_rule = default_inner_rule if in_end == 'Inner'\
            else default_direct_rule if in_end == 'Direct'\
            else default_outer_rule
            seg['iseg'] = iseg_rule
        if '-odefault' in option:
            oseg_rule = default_inner_rule if in_end == 'Inner'\
            else default_direct_rule if in_end == 'Direct'\
            else default_outer_rule
            seg['oseg'] = oseg_rule
        if '-mdefault' in option:
            mseg_rule = default_middle_rule
            seg['mseg'] = mseg_rule
        seg_cmd_list.append(seg)
# print(seg_rules_dict)
# print(seg_cmd_list)