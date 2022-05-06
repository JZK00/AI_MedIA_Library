n = 100

sum = 0
counter = 1
while counter <= n:
    sum = sum + counter
    counter += 1

print(f"1 到 {n} 之和为: {sum}")
print("1 到 {} 之和为: {}".format(n, sum))
print("1 到 %d 之和为: %d" % (n, sum))
