# This is the utility function file

from parse_cmd import seg_rules_dict, seg_cmd_list

import klayout.db as db
import random
from PIL import Image
import numpy as np
import os

# Prepare gds files
def prepare_data(file_name):
    layout = db.Layout()
    layout.dbu = 0.001
    top_cell = layout.create_cell("TOP")
    layer = layout.layer(1,0)
    shapes = top_cell.shapes(layer)
    poly = db.DPolygon([
      db.DPoint(0, 0), db.DPoint(0, 5), db.DPoint(4, 5), db.DPoint(4, 4),
      db.DPoint(1, 4), db.DPoint(1, 3), db.DPoint(3, 3), db.DPoint(3, 2),
      db.DPoint(1, 2), db.DPoint(1, 0)
    ])
    shapes.insert(poly)
    layout.write(file_name)

def move_outer(edge, distance):
    # Judge the direction of the edge
    # (1) to right, (1,0)
    if edge.y1 == edge.y2 and edge.x1 < edge.x2:
        # move upwards
        moved_edge = edge.moved(0, distance)
    # (2) to left, (-1,0)
    elif edge.y1 == edge.y2 and edge.x1 > edge.x2:
        # move downwards
        moved_edge = edge.moved(0, -distance)
    # (3) to up, (0,1)
    elif edge.x1 == edge.x2 and edge.y1 < edge.y2:
        # move rightwards
        moved_edge = edge.moved(-distance, 0)
    elif edge.x1 == edge.x2 and edge.y1 > edge.y2:
        # move leftwards
        moved_edge = edge.moved(distance, 0)
    else:
        raise Exception(f"edge = {str(edge)}, distance = {distance}. Invalid move!")
    return moved_edge

def move_inner(edge, distance):
    return move_outer(edge,-distance)

def exist(point):
    return str(point) != 'None'

def refine_edges(edges):
    # Now edges are in an array
    # Only be used in one shape!
    edges_count = len(edges)
    refined_edges_array=[]
    for i in range(0,edges_count):
        refined_edge = db.Edge()
        # Determine two points of the refined edge
        # Intersection point or cut point, contingent on whether edges are detached
        edge = edges[i]
        previous_edge = edges[(i-1)%edges_count]
        subsequent_edge = edges[(i+1)%edges_count]
        if not edge.is_parallel(previous_edge):
            p1 = edge.intersection_point(previous_edge)
            if not exist(p1):
                p1 = edge.cut_point(previous_edge)
        else:
            p1 = edge.p1
        refined_edge.p1 = p1            
        if not edge.is_parallel(subsequent_edge):
            p2 = edge.intersection_point(subsequent_edge)
            if not exist(p2):
                p2 = edge.cut_point(subsequent_edge)
        else:
            # Insert a new edge
            edge_interted = db.Edge(edge.p2, subsequent_edge.p1)
            p2 = edge.p2
            refined_edges_array.append(edge_interted)
        refined_edge.p2 = p2
        refined_edges_array.append(refined_edge)
    refined_edges = db.Edges(refined_edges_array)
    return refined_edges_array

def edges2edge_array(edges):
    edge_array = []
    for edge in edges.each():
        edge_array.append(edge)
    return edge_array

def get_random_offset(endpoint_a, endpoint_b):
    # Return random float (endpoint_a,endpoint_b)
    return random.randint(endpoint_a, endpoint_b)

def judge_corner(prev_edge, edge, subs_edge, shape):
    # Judge by the close point of the corner
    unit_vec_11 = prev_edge.d()*(-1/prev_edge.length())
    unit_vec_12 = edge.d()*(1/edge.length())
    unit_vec_21 = edge.d()*(-1/edge.length())
    unit_vec_22 = subs_edge.d()*(1/subs_edge.length())
    judge_point_1 = edge.p1 + unit_vec_11 + unit_vec_12
    judge_point_2 = edge.p2 + unit_vec_21 + unit_vec_22
    # if shape.is_box():
    #     in_end = 'Outer' if shape.box.contains(judge_point_1) else 'Inner'
    #     outer_end = 'Outer' if shape.box.contains(judge_point_2) else 'Inner'
    # elif shape.is_polygon():
    #     in_end = 'Outer' if shape.polygon.inside(judge_point_1) else 'Inner'
    #     outer_end = 'Outer' if shape.polygon.inside(judge_point_2) else 'Inner'
    # elif shape.is_path():
    #     in_end = 'Outer' if shape.path.polygon().inside(judge_point_1) else 'Inner'
    #     outer_end = 'Outer' if shape.path.polygon().inside(judge_point_2) else 'Inner'
    # else:
    #     assert False
    assert type(shape) == db.Polygon
    in_end = 'Outer' if shape.inside(judge_point_1) else 'Inner'
    outer_end = 'Outer' if shape.inside(judge_point_2) else 'Inner'
    return in_end, outer_end

def get_lens_by_rule(matched_edges, seg_part, rule, in_end_sep_len = 0, out_end_sep_len = 0):
    # One seg option in command matching one rule, all edges for one shape
    print(f"Deal with {seg_part}, rule is {rule} ...")
    separation_lens_edges = []
    separation_lens = []
    # in_end_sep_len = 0
    # out_end_sep_len = 0
    if 'segnum' in seg_rules_dict[rule] and 'segpara' in seg_rules_dict[rule]:
        for i in range(seg_rules_dict[rule]['segnum']):
            mark = seg_rules_dict[rule]['segpara'][0]
            separation_len =  seg_rules_dict[rule]['segpara'][1]
            sample_point = seg_rules_dict[rule]['segpara'][2]
            # Do not care the mark and the sampling point
            if not separation_len.startswith('%'):
                separation_len = int(int(separation_len)/1000/dbu)  # turn nm to unit
            separation_lens.append(separation_len)  # todo: maybe it is %0
    print("matched_edges:", len(matched_edges))
    for edge in matched_edges:
        print("edge:",str(edge), 'length:', edge.length())
        separation_lens_edge = []
        separation_lens = [int(edge.length()*int(i[1:])/100) if\
            type(i) == str and i.startswith('%') else int(i) for i in separation_lens]
        print(f"separation_lens:{separation_lens}")
        if seg_part == 'wseg':
            seg_num = seg_rules_dict[rule]['segnum']
            basic_seg_len = edge.length()/seg_num
            separation_lens_edge = [int((i+1)*basic_seg_len + separation_lens[i]) for i in range(seg_num-1)]
            # This is from the in-end
        elif seg_part == 'iseg':
            # separation_lens = [int(i) for i in separation_lens]
            for i in range(1, len(separation_lens)+1):
                separation_lens_edge.append(sum(separation_lens[0:i]))
            in_end_sep_len = max(separation_lens)
        elif seg_part == 'oseg':
            # separation_lens = [edge.length()-int(i) for i in separation_lens]
            for i in range(1, len(separation_lens)+1):
                separation_lens_edge.append(edge.length()-sum(separation_lens[0:i]))
            out_end_sep_len = min(separation_lens)
    # Start to process edge
    # for edge in matched_edges:
        elif seg_part == 'mseg':
            if 'eslen' in seg_rules_dict[rule]:  # if no segnum, like in MID_SEG_RULE
                eslen = seg_rules_dict[rule]['eslen']
                for i in range((edge.length()-in_end_sep_len-out_end_sep_len)//eslen-1):
                    separation_len = in_end_sep_len + (i+1) * eslen
                    separation_lens_edge.append(separation_len)
            else:
                assert False  # only support eslen so far
            # print(f"seg={segmentation}")
        else:
            assert False  # only support 4 kinds of seg_part
        separation_lens_edges.append(separation_lens_edge)  # a nested_list
        # print("*****", separation_lens_edges)
    return separation_lens_edges, in_end_sep_len, out_end_sep_len

def dissect_edge(edges, shape):
    print(f"Dissecting edge: {edges} ...")

    # First, try to determine the corner catogory of each
    corner_list = []
    for i in range(edges.count()):
        edge = edges[i]
        prev_edge = edges[(i-1)%edges.count()]
        subs_edge = edges[(i+1)%edges.count()]
        in_end, out_end = judge_corner(prev_edge, edge, subs_edge, shape)
        corner_list.append((in_end, out_end))
    # print(corner_list)
    # Second, deal with commands
    dissected_edges = []
    undissected_edges = edges2edge_array(edges)
    for seg_cmd in seg_cmd_list:
        # Now one command
        print(f"\nNow command is {seg_cmd}")
        # Firstly, determine the corner type, because the judge_corner function
        matched_edges = [edges[i] for i in range(edges.count()) if\
            (seg_cmd['in_end'] in ['Any', corner_list[i][0]]) and (seg_cmd['out_end'] in ['Any', corner_list[i][1]])]
        # Secondly, determine the len
        matched_edges = [i for i in matched_edges if\
            seg_cmd['len'][0]/1000*dbu <= i.length() <= seg_cmd['len'][1]/1000*dbu]
        # matched_edges = []
        undissected_edges = list(set(undissected_edges).difference(matched_edges))  # updated the undissected edges
        # Now edges are prepared for dissection
        dissected_lens = [[]]*len(matched_edges)
        dissected_segs = []
        for seg_part in ['wseg', 'iseg', 'idefault', 'oseg', 'odefault','mseg', 'mdefault']:  # resolved default already
            # This order is important!
            if seg_part in seg_cmd.keys():
                rule = seg_cmd[seg_part]
                # dissect_by_rule(matched_edges, seg_part, rule)
                if seg_part in ['iseg', 'idefault']:
                    dissected_lens_part, in_end_sep_len, _ = get_lens_by_rule(matched_edges, seg_part, rule)
                elif seg_part in ['oseg', 'odefault']:
                    dissected_lens_part, _ , out_end_sep_len = get_lens_by_rule(matched_edges, seg_part, rule)
                elif seg_part in ['mseg', 'mdefault']:
                    dissected_lens_part, _, _ = get_lens_by_rule\
                        (matched_edges, seg_part, rule, in_end_sep_len, out_end_sep_len)
                else:
                    # print("in_end_sep_len, out_end_sep_len", in_end_sep_len, out_end_sep_len)
                    dissected_lens_part, _, _ = get_lens_by_rule(matched_edges, seg_part, rule)
                print(f"dissected_lens_part = {dissected_lens_part}")
                dissected_lens = [sorted(dissected_lens[i]+dissected_lens_part[i]) for i in range(len(matched_edges))]
                print(f"dissected_lens = {dissected_lens}")
        for i in range(len(matched_edges)):
            edge = matched_edges[i]
            start_point = edge.p1
            dissected_len_all = dissected_lens[i]  # dissected length
            dissected_len_all.insert(0,0)
            dissected_len_all.append(edge.length())
            vec_unit = edge.d()*float(1/edge.length())
            # dissected_segs = []
            for j in range(len(dissected_len_all)-1):
                # print('type', type(dissected_len_all[j+1]))
                p1 = start_point + vec_unit*dissected_len_all[j]
                p2 = start_point + vec_unit*dissected_len_all[j+1]
                seg = db.Edge(p1,p2)
                # print("seg:", seg, "length:", seg.length())
                dissected_segs.append(seg)
            # dissected_edges[j] = dissected_segs
        dissected_edges = dissected_edges + dissected_segs
    return dissected_edges + undissected_edges

def correct_edge(edges):
    # Now edges are in an array
    moved_edges =[]
    for edge in edges:
        offset = get_random_offset(-50, 50)
        moved_edge = move_inner(edge, offset)
        moved_edges.append(moved_edge)
    corrected_edges = refine_edges(moved_edges)
    return corrected_edges

def process_shape(shape):
    print(f"Processing shape: {shape} ...")
    # dissected_edges_array = []
    shapes = db.Shapes()
    shapes.insert(shape)
    # global dbu = shape.layout().dbu
    edges = db.Edges(shapes)  # this is edges for one shape
    # for i in range(edges.count()):
        # dvec_unit = db.DVector(edges[i].p2-edges[i].p1)/edges[i].length()
        # This section needs improving
    dissected_edges_array = dissect_edge(edges, shape)
    corrected_edges_array = correct_edge(dissected_edges_array)
    corrected_edges = db.Edges(corrected_edges_array)
    new_shape = db.EdgeProcessor().simple_merge_e2p(edges2edge_array(corrected_edges), True, True)
    # new_shape = dissected_edges_array
    return new_shape

def merge_shapes(shapes):
    sp = db.ShapeProcessor()
    shape_array = []
    for shape in shapes.each():
        shape_array.append(shape)
    return sp.merge_to_polygon(shape_array, 0, True, True)

def gds2img(polygon, cell_name, i):
    step = 10
    array = np.zeros(shape=(polygon.bbox().width()//step+1, polygon.bbox().height()//step+1), dtype=np.uint8)
    print(array.shape)
    for x in range(polygon.bbox().p1.x, polygon.bbox().p2.x+1, step):
        for y in range(polygon.bbox().p1.y, polygon.bbox().p2.y+1, step):
            if polygon.inside(db.Point(x,y)):
                array[(x-polygon.bbox().p1.x)//step,(y-polygon.bbox().p1.y)//step] = 255
        print("---", x)
    print(array)
    img = Image.fromarray(np.rot90(array))
    if not os.path.exists('img_layout'):
        os.makedirs('img_layout')
    img.save('img_layout/gds2img_'+cell_name+'_'+str(i)+'.png')


def process_layer(cell, layer):
    # print(f"dbu={dbu}")
    global dbu
    dbu = cell.layout().dbu
    layer_shapes = cell.shapes(layer)
    # print(layer_shapes)
    # Should merge the shapes to polygons
    merged_shapes = merge_shapes(layer_shapes)
    layer_shapes.clear()
    print("***********", merged_shapes)
    i=0
    for polygon in merged_shapes:
        new_shape = process_shape(polygon)
        for i in new_shape:
            cell.shapes(layer).insert(i)
        gds2img(polygon, cell.name, i)
        i+=1
        