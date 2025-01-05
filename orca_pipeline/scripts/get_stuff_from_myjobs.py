# +26
if __name__ == "__main__":
    lines = None
    with open("foo", "r") as f:
        lines = f.readlines()
    job_ids = []
    working_dir = []

    for i in range(0, len(lines), 26):

        if "NEB" in lines[2+i]:

            job_ids.append(lines[1+i].split()[3])
            working_dir.append(lines[9+i].split()[-1])

    with open("job_ids.txt", "w") as f:
        for id in job_ids:
            f.write(id + "\n")    # Your code here
    with open("working_dir.txt", "w") as f:
        for dir in working_dir:
            f.write(dir + "\n")    # Your code here
