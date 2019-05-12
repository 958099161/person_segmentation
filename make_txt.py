txtName = "loss_batch.txt"
f=open(txtName, 'a+')
# f=file(txtName, "a+")
for i in range(1,3):

    new_context = "C++" + '\n'
    f.write(new_context)
f.close()