from pywebcopy import save_webpage
url = 'http://web.archive.org/web/20200610120603/https://www.scotiabank.com/ca/en/personal.html'
kwargs = {'bypass_robots': True}
download_folder = 'temp/'
save_webpage(url, download_folder, **kwargs)