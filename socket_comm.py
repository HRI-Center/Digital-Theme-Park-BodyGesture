from socket import *
import threading

global recvdata
global senddata

class comm:
    def __init__(self):
        self.clientSock = socket(AF_INET, SOCK_STREAM)
        self.clientSock.connect(('127.0.0.1', 3000))
        #self.clientSock = socket(AF_INET, SOCK_STREAM)
        # self.serverSock.bind(('192.168.11.220', 2000))
        #self.serverSock.bind(('127.0.0.1', 3000))
        #self.serverSock.listen(1)
        # self.connectionSock, self.addr = self.serverSock.accept()
        # print(str(self.addr),' connected')

    def send(self,sock):
        while True:
            senddata = 'ok'
            sock.send(senddata.encode('utf-8'))
            print('전송완료')
            if senddata == '/quit':
                print('연결정상종료')
                break

    def receive(self, sock):
        while True:
            recvdata = sock.recv(1024)
            if not recvdata:
                print('no receive data')
                sock.close()
                break
            print('받은 데이터:', recvdata.decode('utf-8'))
            self.send_func(sock)

    def send_func(self, sock):
        temp = input("전송대기중....")
        sock.send(senddata.encode('utf-8'))
        print('전송완료')

    def run(self):
        receiver = threading.Thread(target=self.receive, args=(self.clientSock,))
        receiver.daemon = True
        receiver.start()




