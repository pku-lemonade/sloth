global_message_id = 0
global_op_id = 0

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

pe_instruction = [[] for _ in range(16)]

def solve(id, l, r):
    print(f"{l}, {r}")
    if l == r:
        if l % 2 == 0:
            pe_instruction[l].append(read("POOL", "FEAT", 32, 32-id))
            pe_instruction[l].append(comp("POOL", 0, 32-id))
        else:
            pe_instruction[l].append(read("CONV", "FEAT", 32, 32-id))

        return

    mid = (l + r) // 2
    solve(id * 2, l, mid)
    solve(id * 2 + 1, mid+1, r)

    pe_instruction[mid].append(send(r, "CONV", "PARA", 32, 32-id))
    pe_instruction[r].append(recv(mid, "CONV", "PARA", 32, 32-id))

    pe_instruction[r].append(comp("CONV", 32, 32-id))
    pe_instruction[r].append(stay("CONV", 32, 32-id))

solve(1, 0, 15)
pe_instruction[15].append(write("CONV", 32, 31))

with open("tests/fab_tree/instructions.txt", "w") as file:

    for id in range(16):
        print(f"PE {id} {{", file=file)
        for inst in pe_instruction[id]:
            print(f"    {inst}", file=file)
        print("}", file=file)