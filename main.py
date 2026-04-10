import imagingcontrol4 as ic4
import multiprocessing as mp
import cv2
from ultralytics import YOLO
import time
import numpy as np


def configCam():
    """
    配置相机

    """
    ic4.Library.init()
    deviceList = ic4.DeviceEnum.devices()
    device_info = deviceList[0]
    print(f'找到相机：{device_info}')

    # 打开相机
    grabber = ic4.Grabber()
    grabber.device_open(device_info)

    # 属性映射
    propertyMap = grabber.device_property_map
    # 设定帧像素格式为:BayerRG8
    propertyMap.set_value(ic4.PropId.PIXEL_FORMAT, ic4.PixelFormat.BayerRG8)
    propertyMap.set_value(ic4.PropId.WIDTH, 720)
    propertyMap.set_value(ic4.PropId.HEIGHT, 900)

    # 设定采集帧率
    fps = 20
    propertyMap.set_value(ic4.PropId.ACQUISITION_FRAME_RATE, fps)

    # 设定曝光时间
    propertyMap.set_value(ic4.PropId.EXPOSURE_AUTO, "Off")
    propertyMap.set_value(ic4.PropId.EXPOSURE_TIME, 5000.0)  # 500微秒
    
    class Listener(ic4.QueueSinkListener):
        def sink_connected(self, sink: ic4.QueueSink, image_type: ic4.ImageType, min_buffers_required: int) -> bool:
        # Allocate more buffers than suggested, because we temporarily take some buffers
        # out of circulation when saving an image or video files.
            sink.alloc_and_queue_buffers(min_buffers_required + 30)   # min_buffers_required=6
            return True

        def sink_disconnected(self, sink: ic4.QueueSink):
            pass
        
        # 在单独的线程中执行
        def frames_queued(self, sink: ic4.QueueSink):
            buf = sink.pop_output_buffer()
            nparray = buf.numpy_copy()
            bgr_img = cv2.cvtColor(nparray, cv2.COLOR_BayerBG2BGR)
            if g_imgQueue.qsize() > 70:
                print(f'g_imgQueue.qsize()={g_imgQueue.qsize()} from listener. 请降低帧率')
            try:
                g_imgQueue.put_nowait(bgr_img)
            except Exception:
                pass
            
            buf.release()
        
    sink = ic4.QueueSink(Listener())
    grabber.stream_setup(sink=sink, setup_option=ic4.StreamSetupOption.DEFER_ACQUISITION_START)
    print('相机配置完成')
    return grabber

def YOLO_process(imgQ_in: mp.Queue, imgQ_d: mp.Queue, e_acq, e_exit):
    """
        imgQ_in: 相机采集到的图像队列
        imgQ_d: YOLO处理后的,在主进程中显示的图像队列
    """
    print("-------YOLO_process started-------", k)
    # 载入模型
    model = YOLO("./models/1/best1-fp16.engine", task="detect")
    # 通知主进程开始采集图像
    e_acq.set()
    # 从队列中获取一张图像(阻塞操作)
    img = imgQ_in.get()     
    # 一旦获取到图像，就可以通知主进程关闭采集
    e_acq.set()

    # YOLO模型预热 计算预热时间
    start = time.perf_counter()
    # 预热模型,进行一次预测,让YOLO加载模型文件
    model.predict(img, half=True, verbose=True, device=0)
    end = time.perf_counter()
    print(f"模型预热时间: {(end-start):.3f} 秒")
    img_h, img_w, _ = img.shape
    
    # 清空imgQ_in队列, 保证里面只有最新的图像, 避免图像延迟
    while True:
        try:
            imgQ_in.get_nowait()
        except:
            break
    
    print(f'是否清空队列？imgQ_in.qsize()={imgQ_in.qsize()}')
    e_acq.set()         # 通知主进程开始采集图像

    # 检测到的两粒种子像素距离列表
    dist_q = []

    # 计算得到的粒距列表
    liju = []

    while True:
        # 检测e_exit信号，如果被设置，退出while循环
        if e_exit.is_set():
            print("YOLO进程收到退出信号，YOLO进程正在退出...")
            # 退出前清空队列
            while True:
                try:
                    imgQ_in.get_nowait()
                except:
                    break
            while True:
                try:
                    imgQ_d.get_nowait()
                except:
                    break
            # 退出最外层的while循环,结束YOLO进程
            break

        # 当imgQ_in队列中的图像产生积压(超过50张)时，只显示图像，而不对帧进行推理。
        while imgQ_in.qsize() > 50:
            try:
                imgQ_d.put_nowait(imgQ_in.get_nowait())
            except:
                pass
        
        try:
            # 非阻塞获取图像
            img = imgQ_in.get_nowait()
        except:
            # 如果imgQ_in队列为空，继续下一个循环
            continue
        
        result = model.predict(img, conf=0.8, half=True, verbose=True, device=0)[0]
        boxes = result.boxes
        # 未检测到种子
        if boxes is None or len(boxes) == 0:
            print("没有目标",len(boxes))
        # 在图像上检测到两粒种子
        if len(boxes) == 2:
            centers = boxes.xywh.cpu().numpy()[:,:2]
            cy = centers[:,1]
            dist = abs(cy[0]-cy[1])
            # 如果两粒种子的y坐标差值小于img高的一半，说明前后两粒种子离得太近了，略过这一帧
            if dist < (img_h // 2):
                continue
            dist_q.append(dist)
        # 当在图像上检测到一粒种子时
        if len(boxes) == 1:
            if len(dist_q) == 0:  # 如果 dist_q 里没有数据
                continue
            else:
                # 如果 dist_q 里有数据, 计算数据的平均值作为前一个粒距数据，并清空 dist_q
                dist_arr = np.array(dist_q)
                dist_mean = np.mean(dist_arr)
                liju.append(k * dist_mean)
                dist_q.clear()
                print("[-----粒距列表-----]：", liju)

        try:
            # 将检测后的图像放入显示队列
            imgQ_d.put_nowait(result.plot())
        except:
            continue

    print(imgQ_in.qsize(), imgQ_d.qsize())
#    imgQ_d.close()
#    imgQ_in.close()
    print("YOLO进程已退出.")

if __name__ == '__main__':
    g_imgQueue  = mp.Queue(maxsize=100)          # 相机采集的图像队列
    g_dispQueue = mp.Queue(maxsize=100)          # 处理后的图像队列（在窗口上显示）
    e_acq = mp.Event()                           # 事件对象
    e_exit = mp.Event()                          # 进程退出信号
    
    # 配置相机  
    grabber = configCam()

    # 相机标定系数 cm/pixel(每像素多少厘米)
    k = 0.5
    
    # 启动YOLO检测进程
    p_yolo = mp.Process(target=YOLO_process, args=(g_imgQueue, g_dispQueue, e_acq, e_exit))
    p_yolo.start() 
    
    # 主进程阻塞，等待YOLO进程将e_acq设置为True
    e_acq.wait()
    
    # e_acq设置为True后，相机开始采集图像
    e_acq.clear() # 清除事件状态标志,变为False
    grabber.acquisition_start()
    print('-------相机开始采集-------')

    # 等待YOLO进程发出"停止采集"信号
    e_acq.wait()
    e_acq.clear() # 清除事件状态标志,变为False
    grabber.acquisition_stop()
    print('-------相机停止采集-------')

    e_acq.wait() # 等待YOLO进程给出开始采集的信号
    e_acq.clear()
    grabber.acquisition_start()
    print('-------相机开始采集-------')

    cv2.namedWindow('video')
    while True:
        try:
            img = g_dispQueue.get_nowait()
        except:
            continue
        cv2.imshow('video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            e_exit.set() # 发送退出信号给YOLO进程
            grabber.acquisition_stop()
            grabber.device_close()
            cv2.destroyAllWindows()
            break

#    p_yolo.join(timeout=2)  # 等待子进程退出
#    if p_yolo.is_alive():
#        print("YOLO进程未能正常退出。")
    print("主进程已退出.")
    


            


