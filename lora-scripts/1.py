import os

# 定义要克隆的仓库列表和相应的目录名
repositories = [
    ("https://gitcode.net/overbill1683/stablediffusion", "stable-diffusion-stability-ai"),
    ("https://gitcode.net/overbill1683/taming-transformers", "taming-transformers"),
    ("https://gitcode.net/overbill1683/k-diffusion", "k-diffusion"),
    ("https://gitcode.net/overbill1683/CodeFormer", "CodeFormer"),
    ("https://gitcode.net/overbill1683/BLIP", "BLIP")
]

def update_repositories():
    workspace_path = "/mnt/workspace/stable-diffusion-webui"
    
    # 切换到工作目录
    os.chdir(workspace_path)
    
    # 更新主仓库
    os.system("git pull")
    
    # 检查并更新子仓库
    os.chdir(os.path.join(workspace_path, "repositories"))
    for repo_url, repo_dir in repositories:
        repo_path = os.path.join(workspace_path, "repositories", repo_dir)
        
        if os.path.exists(repo_path):
            # 切换到子仓库目录
            os.chdir(repo_path)
            
            # 获取远程仓库的最新信息
            os.system("git fetch")
            
            # 检查本地和远程仓库的差异
            diff_output = os.popen("git diff HEAD..origin/master").read().strip()
            
            if diff_output:
                # 有更新，拉取最新代码
                os.system("git pull")
                print(f"子仓库 {repo_dir} 更新成功！")
            else:
                print(f"子仓库 {repo_dir} 已经是最新的，无需更新。")
        else:
            # 子仓库不存在，克隆仓库
            os.chdir(workspace_path)
            os.system(f"git clone {repo_url} {repo_dir}")
            print(f"成功克隆子仓库 {repo_dir}！")
    
    # 下载配置文件
    os.system("wget -O config.json https://gitcode.net/Akegarasu/sd-webui-configs/-/raw/master/config.json")
    print("配置文件下载成功！")

if __name__ == "__main__":
    update_repositories()
    print("主仓库和子仓库更新完成。")
