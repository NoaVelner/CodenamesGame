def create_results_list_from_file(path_to_results):
    results_list = []
    with open(path_to_results) as infile:
        for i, line in enumerate(infile):
            line = line.rstrip().split(' ')
            if len(line) <= 1:
                return results_list
            if i == 0:
                results_list.append(line[1][1:-4])
            results_list.append([int(line[3][:-1]), int(line[5][:-1]), int(line[7][:-1]) + int(line[9][:-1]), int(line[11][:-1])])
    return results_list

if __name__ == "__main__":
    print(create_results_list_from_file('results/bot_results_new_style.txt'))