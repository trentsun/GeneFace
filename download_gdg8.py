import requests

def download_file_from_onedrive(url, destination):
    # OneDrive直链转换
    data = requests.get(url)
    download_url = data.url.replace('redir', 'download')
    
    # 下载文件
    resp = requests.get(download_url)
    with open(destination, 'wb') as f:
        f.write(resp.content)

# 使用示例
shared_link = 'https://1drv.ms/v/s!AlimBn4it8JBiyuVfAYgvd3djqLQ?e=5fkHLo'
download_file_from_onedrive(shared_link, 'gdg8.mp4')