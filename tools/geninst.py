import re

filename = "tools/mapping.txt"

border_pattern = r'\d+'
pipeline_pattern = r'(\d+)'
mapping_pattern = r'(\d+),(\d+),(\d+),(\d+)'

global_message_id = 0

def read(type, tag, read_size, id):
    global global_message_id 
    global_message_id += 1
    return f"READ {global_message_id} {type} {id} {tag} -1 {read_size}"

def write(type, write_size, id):
    global global_message_id 
    global_message_id += 1
    return f"WRITE {global_message_id} {type} {id} WGT -1 {write_size}"

def send(dst, type, tag, size, id):
    global global_message_id 
    global_message_id += 1
    return f"SEND {global_message_id} {type} {id} {tag} {dst} {size}"

def recv(src, type, tag, size, id):
    return f"RECV {global_message_id} {type} {id} {tag} {src} {size}"

def comp(type, size, id):
    global global_message_id 
    global_message_id += 1
    return f"COMP {global_message_id} {type} {id} FEAT -1 {size}"

def stay(type, size, id):
    global global_message_id 
    global_message_id += 1
    return f"STAY {global_message_id} {type} {id} FEAT -1 {size}"

def to1d(x, y):
    return x*4 + y

# resnet各层特征图大小
resnet50 = [
    # layer0
    150528, 802816, 
    # layer1
    200704, 200704, 802816, 802816, 802816, 200704, 200704, 802816, 802816, 200704, 200704, 802816, 802816, 
    # layer2
    100352, 100352, 401408, 401408, 401408, 100352, 100352, 401408, 401408, 100352, 100352, 401408, 401408, 100352, 100352, 401408, 401408,
    # layer3
    50176, 50176, 200704, 200704, 200704, 50176, 50176, 200704, 200704, 50176, 50176, 200704, 200704, 50176, 50176, 200704, 200704, 50176, 50176, 200704, 200704, 50176, 50176, 200704, 200704, 
    # layer4
    25088, 25088, 100352, 100352, 100352, 25088, 25088, 100352, 100352, 25088, 25088, 100352, 100352, 
    # pool&fc
    100352, 2048000
]

# resnet各层参数量
resnet50para = [
    # layer0
    9408, 0,
    # layer1
    4096, 36864, 16384, 16384, 16384, 16384, 36864, 16384, 16384, 16384, 36864, 16384, 16384,
    # layer2
    32768, 147456, 65536, 131072, 131072, 65536, 147456, 65536, 131072, 65536, 147456, 65536, 131072, 65536, 147456, 65536, 131072,
    # layer3
    131072, 589824, 262144, 524288, 524288, 262144, 589824, 262144, 524288, 262144, 589824, 262144, 524288, 262144, 589824, 262144, 524288, 262144, 589824, 262144, 524288, 262144, 589824, 262144, 524288,
    # layer4
    524288, 2359296, 1048576, 2097152, 2097152, 1048576, 2359296, 1048576, 2097152, 1048576, 2359296, 1048576, 2097152,
    # pool&fc
    0, 2048000
]

resnet50compute = [
    # layer0
    118013952, 1806336,
    # layer1
    12845056, 115605504, 462422016, 51380224, 802816, 51380224, 115605504, 462422016, 802816, 51380224, 115605504, 462422016, 802816,
    # layer2
    25690112, 115605504, 51380224, 102760448, 401408, 51380224, 115605504, 51380224, 401408, 51380224, 115605504, 51380224, 401408, 51380224, 115605504, 51380224, 401408,
    # layer3
    25690112, 115605504, 51380224, 102760448, 200704, 51380224, 115605504, 51380224, 200704, 51380224, 115605504, 51380224, 200704, 51380224, 115605504, 51380224, 200704, 51380224, 115605504, 51380224, 200704, 51380224, 115605504, 51380224, 200704,
    # layer4
    25690112, 115605504, 51380224, 102760448, 100352, 51380224, 115605504, 51380224, 100352, 51380224, 115605504, 51380224, 100352,
    # pool&fc
    100352, 2048000
]

resnet50type = [
    # layer0
    "CONV", "POOL",
    # layer1
    "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV",
    # layer2
    "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV",
    # layer3
    "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV",
    # layer4
    "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", "CONV", 
    # pool&fc
    "POOL", "FC"
]

# 16个PE
layer_partition = [[] for _ in range(72)]
pe_instruction = [[] for _ in range(16)]

with open(filename, "r", encoding="utf-8") as file:
    pipeline_border = []
    pipeline_begin = 0
    pipeline_end = 0
    pipeline_counter = 0

    linecount = 0

    for line in file:
        linecount += 1
        if line.find("border") != -1:
            numbers = re.findall(border_pattern, line)
            pipeline_border = [int(x) for x in numbers]
            continue

        if line.find("pipeline") != -1:
            pipeline_counter = 0
            pid = re.search(pipeline_pattern, line)
            pid = pid.group()
            pid = int(pid)
            if pid == 1:
                pipeline_begin = 0
                pipeline_end = pipeline_border[pid-1]
            elif pid == len(pipeline_border)+1:
                pipeline_begin = pipeline_border[pid-2] + 1 
                pipeline_end = 71
            else:
                pipeline_begin = pipeline_border[pid-2] + 1
                pipeline_end = pipeline_border[pid-1]
            continue
        
        mappings = re.search(mapping_pattern, line)
        x, y, lid, partid = mappings.groups()
        x, y, lid, partid = int(x), int(y), int(lid), int(partid)
        # 维护第lid个layer的mapping
        layer_partition[lid].append((x, y))
        pipeline_counter += 1
        
        # 一个pipeline的mapping读入完毕
        if pipeline_counter == 16:
            print(f"[{pipeline_begin}, {pipeline_end}]")
            # pipeline初始读入
            for part in layer_partition[pipeline_begin]:
                # 第一层读入
                if pipeline_begin == 0:
                    pe_instruction[to1d(part[0], part[1])].append(read(resnet50type[pipeline_begin], "PARA", resnet50para[pipeline_begin] // len(layer_partition[pipeline_begin]), pipeline_begin))
                    pe_instruction[to1d(part[0], part[1])].append(read(resnet50type[pipeline_begin], "FEAT", resnet50[pipeline_begin] // len(layer_partition[pipeline_begin]), pipeline_begin))
                    pe_instruction[to1d(part[0], part[1])].append(comp(resnet50type[pipeline_begin], resnet50compute[pipeline_begin] // len(layer_partition[pipeline_begin]), pipeline_begin))
                    
            
            # pipeline数据发送
            if pipeline_begin > 0:
                for id in range(pipeline_begin, pipeline_end+1):
                    src_size = resnet50[id] // len(layer_partition[id-1])
                    dst_size = resnet50[id] // len(layer_partition[id])
                    # 记录发送端part位置和余量
                    src_pos = 0
                    src_remain = src_size
                    # 枚举接收端
                    for part in layer_partition[id]:
                        # 读入权重
                        pe_instruction[to1d(part[0], part[1])].append(read(resnet50type[id], "PARA", resnet50para[id], id))
                        dst_remain = dst_size
                        while dst_remain > 0:
                            cost = min(src_remain, dst_remain)
                            if cost > 0:
                                if layer_partition[id-1][src_pos] != part:
                                    # 读入特征，发送与接收对应
                                    pe_instruction[to1d(layer_partition[id-1][src_pos][0], layer_partition[id-1][src_pos][1])].append(send(to1d(part[0], part[1]), resnet50type[id], "FEAT", cost, id-1))
                                    pe_instruction[to1d(part[0], part[1])].append(recv(to1d(layer_partition[id-1][src_pos][0], layer_partition[id-1][src_pos][1]), resnet50type[id], "FEAT", cost, id))
                                    # 每条信息有唯一id
                                    # global_message_id += 1
                                else:
                                    pe_instruction[to1d(part[0], part[1])].append(stay(resnet50type[id], dst_size, id))
                            src_remain -= cost
                            dst_remain -= cost
                            if src_remain == 0:
                                src_pos += 1
                                src_remain = src_size

                            if src_pos >= len(layer_partition[id-1]):
                                break

                        pe_instruction[to1d(part[0], part[1])].append(comp(resnet50type[id], resnet50compute[id] // len(layer_partition[id]), id))
                        if src_pos >= len(layer_partition[id-1]):
                            break

            # pipeline结束写入
            for part in layer_partition[pipeline_end]:
                pe_instruction[to1d(part[0], part[1])].append(write(resnet50type[pipeline_end], resnet50[pipeline_end] // len(layer_partition[pipeline_end]), pipeline_end))

with open("tools/instructions.txt", "w") as file:
    for id in range(16):
        print(f"PE {id} {{", file=file)
        for inst in pe_instruction[id]:
            print(f"    {inst}", file=file)
        print("}", file=file)