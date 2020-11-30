# import os

# # check for empty images

# names = os.listdir(dr)

# paths = [os.path.join(dr, name) for name in names]

# sizes = [(path, os.stat(path).st_size) for path in paths]

# empties = 0
# for s in sizes:
#   if s[1] < 100:
#     empties =+ 1


# # change requirements.txt so that newest versions are installed

# sep = "="
# req = []
# with open("multi-label-soft-f1/requirements_gpu.txt", "r") as f:
#    for l in f.readlines():
#          req.append(l.replace("==", ">=")) #l.split(sep, 1)[0]
# req.append("keras")
# req.append("tensorflow")
# with open("multi-label-soft-f1/my_req_gpu.txt", "w") as f:
#   for r in req:
#     f.write("%s\n" % r)
