from SoccerNet.Downloader import SoccerNetDownloader


d = SoccerNetDownloader(LocalDirectory="datasets/SoccerNet")
d.downloadDataTask(task="tracking", split=["train","test"])