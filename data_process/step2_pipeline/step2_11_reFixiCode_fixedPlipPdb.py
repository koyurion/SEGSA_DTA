import os

def get_ori_dict(pdb_name):
    ori_dict = dict()
    crash_COT_flag = "0"
    ori_pdb = ori_pdb_path + pdb_name + "_delHydrogenWithHOH.pdb"
    for line in open(ori_pdb).readlines():
        '''
        if line[22:27].strip() == "0":
            print("error: ori resNum = 0 exist!")
            return ori_dict, "1"
        '''
        if line.startswith("ATOM") or line.startswith("HETATM"):
            if line[26:27] != " " or line[22:27].strip() == "0":
                coorOccupTemp = line[30:66]
                val = ori_dict.get(coorOccupTemp,"empty")
                if val == "empty":
                    ori_dict[coorOccupTemp] = line[22:27]
                else:
                    crash_COT_flag = "2"
                    return ori_dict,crash_COT_flag

    return ori_dict,crash_COT_flag

ori_pdb_path = "step1_screen/94_4_delHydrogenWithHOH/"
fixedPdb_path = "step1_screen/after_plip_pdbs_0607/"
out_path = "step1_screen/reFixiCode_plip_pdbs_0607/"

fixedPdbs = [i for i in os.listdir(fixedPdb_path) if i.endswith("_fixedPlip.pdb")]
crash_COT_list = []
oriResNum0_list =[]
miss_COT_list = []
count = 0
done_count = 0
for fp in fixedPdbs:
    count += 1
    pdb_name,afteprex = fp.split("_")
    print(count,pdb_name)

    ori_dict,crash_COT_flag = get_ori_dict(pdb_name)
    if crash_COT_flag == "0":
        pass
    elif crash_COT_flag == "1":
        oriResNum0_list.append(pdb_name)
        continue
    elif crash_COT_flag == "2":
        crash_COT_list.append(pdb_name)
        continue
    else:
        assert False

    out_file = out_path + pdb_name + "_reFixiCode.pdb"
    out_file = open(out_file, "w")
    text = open(fixedPdb_path + pdb_name + "_" + afteprex).readlines()
    for line in text:
        if (line.startswith("ATOM") or line.startswith("HETATM")) and line[22:27].strip() == "0":
            # print("ori",line)
            coorOccupTemp = line[30:66]
            resName = ori_dict.get(coorOccupTemp,"empty")
            if resName != "empty":
                newline = line[:22] + resName + line[27:]
            else:
                newline = line
                miss_COT_list.append(pdb_name)
            # print("new",newline)
            out_file.write(newline)
        else:
            out_file.write(line)
    done_count += 1

print("crash_COT_list",len(crash_COT_list),crash_COT_list)
print("oriResNum0_list",len(oriResNum0_list),oriResNum0_list)
print("miss_COT_list",len(miss_COT_list),miss_COT_list)
print("count", count,"done count", done_count)