sites = ["Baidu", "Google", "ai", "Taobao"]
for site in sites:
    if site == "ai":
        print("AI教程!")
        break
    print("循环数据 " + site)
else:
    print("没有循环数据!")
print("完成循环!")
