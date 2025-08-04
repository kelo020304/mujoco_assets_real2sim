import sys

def print_ply_fields(ply_path):
    with open(ply_path, 'rb') as f:  # 用二进制方式打开
        header = []
        while True:
            line = f.readline()
            decoded = line.decode("ascii", errors="ignore").strip()
            header.append(decoded)
            if decoded == "end_header":
                break

    print("======== PLY Header Fields ========")
    for line in header:
        if line.startswith("property"):
            print(line)
    print("===================================")

    # 统计 f_dc, f_rest 字段数量
    f_dc_count = sum("f_dc" in line for line in header)
    f_rest_count = sum("f_rest" in line for line in header)

    print(f"\nTotal f_dc_* fields: {f_dc_count}")
    print(f"Total f_rest_* fields: {f_rest_count}")
    total_feat = f_dc_count + f_rest_count
    if total_feat > 0:
        est_sh_degree = int(((total_feat + 3) / 3)**0.5) - 1
    else:
        est_sh_degree = 0
    print(f"Estimated SH degree: {est_sh_degree}")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python print_ply_fields.py path_to_ply_file")
        sys.exit(1)

    ply_path = sys.argv[1]
    print_ply_fields(ply_path)
