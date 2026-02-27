from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import astrbot.api.message_components as Comp
import os
import cv2
import hyperlpr3 as hyp3
import json
import subprocess
import asyncio
import datetime
import uuid  # 用于生成唯一文件名
import aiohttp  # 异步请求下载网络图片
import base64
import easyocr
from paddleocr import PaddleOCR
'''相对路径原点是astrbot的main.py'''

@register("6343bot_cha_cha_bus", "cha_cha_bus", "查询公交车牌、线路信息", "0.2.0")
class ChaChaBus(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.waiting_users = {} # 等待用户输入的用户列表 用于车牌识别功能

    async def initialize(self):
        """可选择实现异步的插件初始化方法，当实例化该插件类之后会自动调用该方法。"""

    # 查车牌功能
    @filter.command("ccp", alias={'查车牌', '查车', 'ccb'})
    async def ccp(self, event: AstrMessageEvent, bus_id: str, city: str = ''):
        """查询公交车牌功能""" # 这是 handler 的描述，将会被解析方便用户了解插件内容。建议填写。
        user_name = event.get_sender_name()
        message_str = event.message_str # 用户发的纯文本消息字符串
        message_chain = event.get_messages() # 用户所发的消息的消息链 # from astrbot.api.message_components import *
        logger.info(message_chain)
        found = 0
        bus_id = bus_id.upper()
        '''下面是指令核心部分'''

        if city == '' or city == '深圳' or city == 'SZ' or city == 'sz' or city == 'B' or city == 'b':
            with open('../cha cha bus/cha-cha-bus json/Shenzhen Bus List.json', 'r', encoding='utf-8') as f:
                shenzhen_bus_dict = json.load(f)
                datadate = shenzhen_bus_dict['date']

            # 尝试找5位数+‘D’
            try :
                bus_id_with_D = bus_id + 'D'
                bus = shenzhen_bus_dict[bus_id_with_D]
                output_str = ''
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自深圳交通百科\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                found = 1
            except:
                pass

            # 尝试找原有输入
            try :
                bus = shenzhen_bus_dict[bus_id]
                output_str = ''
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自深圳交通百科\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                found = 1
            except:
                pass

            if found == 1 :
                return

        if city == '' or city == '广州' or city == 'GZ' or city == 'gz' or city == 'A' or city == 'a':
            with open('../cha cha bus/cha-cha-bus json/Guangzhou Bus List.json', 'r', encoding='utf-8') as f:
                guangzhou_bus_dict = json.load(f)
                datadate = guangzhou_bus_dict['date']

            # 尝试找原输入
            try :
                bus = guangzhou_bus_dict[bus_id]
                if '所属线路' not in bus:
                    long_bus_id = bus['车牌']
                    bus_id = long_bus_id[2:]
                    bus = guangzhou_bus_dict[bus_id]

                output_str = f'车牌: 粤A {bus_id}\n'
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自广州交通维基\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                return
            except:
                pass

            # 尝试找5位数+‘D’
            try :
                bus_id_with_D = bus_id + 'D'
                bus = guangzhou_bus_dict[bus_id_with_D]
                output_str = f'车牌: 粤A {bus_id_with_D}\n'
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自广州交通维基\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                return
            except:
                pass

            # 尝试找5位数+‘F’
            try :
                bus_id_with_F = bus_id + 'F'
                bus = guangzhou_bus_dict[bus_id_with_F]
                output_str = f'车牌: 粤A {bus_id_with_F}\n'
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自广州交通维基\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                return
            except:
                pass

            # 尝试加'-'
            for i in range(len(bus_id)-1):
                try :
                    bus_id_with_dash = bus_id[:i+1] + '-' + bus_id[i+1:]
                    bus_id = guangzhou_bus_dict[bus_id_with_dash]['车牌'][2:]
                    bus = guangzhou_bus_dict[bus_id]
                    output_str = f'车牌: 粤A {bus_id}\n'
                    for key in bus:
                        output_str += f'{key}: {bus[key]}\n'
                    output_str += f'以上信息来自广州交通维基\n数据日期: {datadate}'
                    yield event.plain_result(output_str)
                    return
                except:
                    pass

        if city == '' or city == '珠海' or city == 'ZH' or city == 'zh' or city == 'C' or city == 'c':
            with open('../cha cha bus/cha-cha-bus json/Zhuhai Bus List.json', 'r', encoding='utf-8') as f:
                zhuhai_bus_dict = json.load(f)
                datadate = zhuhai_bus_dict['date']

            # 尝试找5位数+‘D’
            try :
                bus_id_with_D = bus_id + 'D'
                bus = zhuhai_bus_dict[bus_id_with_D]
                output_str = f'车牌: 粤C {bus_id_with_D}\n'
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自珠海交通维基\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                found = 1
            except:
                pass

            # 尝试找原有输入
            try :
                bus = zhuhai_bus_dict[bus_id]
                output_str = f'车牌: 粤C {bus_id}\n'
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自珠海交通维基\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                found = 1
            except:
                pass

            if found == 1 :
                return
            
        if city == '' or city == '佛山' or city == 'FS' or city == 'fs' or city == 'E' or city == 'e' or city == 'X' or city == 'x' or city == 'Y' or city == 'y':
            with open('../cha cha bus/cha-cha-bus json/Foshan Bus List.json', 'r', encoding='utf-8') as f:
                foshan_bus_dict = json.load(f)
                datadate = foshan_bus_dict['date']

            # 尝试找原输入
            try :
                bus = foshan_bus_dict[bus_id]
                if '所属线路' not in bus: 
                    bus_id = bus['车牌']
                    bus = foshan_bus_dict[bus_id]

                output_str = f'车牌: {bus_id}\n'
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自佛山公交百科\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                return
            except:
                pass

            # 尝试找‘粤E’+5位数+‘D’
            try :
                bus_id_E_D = '粤E'+ bus_id + 'D'
                bus = foshan_bus_dict[bus_id_E_D]
                output_str = f'车牌: {bus_id_E_D}\n'
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自佛山公交百科\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                return
            except:
                pass

            # 尝试找‘粤E’+5位数+‘F’
            try :
                bus_id_E_F = '粤E'+ bus_id + 'F'
                bus = foshan_bus_dict[bus_id_E_F]
                output_str = f'车牌: {bus_id_E_F}\n'
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自佛山公交百科\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                return
            except:
                pass

            # 尝试找‘粤E’+5位数
            try :
                bus_id_E = '粤E'+ bus_id
                bus = foshan_bus_dict[bus_id_E]
                output_str = f'车牌: {bus_id_E}\n'
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自佛山公交百科\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                return
            except:
                pass

            # 尝试找‘粤X’+5位数
            try :
                bus_id_X = '粤X'+ bus_id
                bus = foshan_bus_dict[bus_id_X]
                output_str = f'车牌: {bus_id_X}\n'
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自佛山公交百科\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                return
            except:
                pass

            # 尝试找‘粤Y’+5位数
            try :
                bus_id_Y = '粤Y'+ bus_id
                bus = foshan_bus_dict[bus_id_Y]
                output_str = f'车牌: {bus_id_Y}\n'
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自佛山公交百科\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                return
            except:
                pass

            # 尝试加'-'
            for i in range(len(bus_id)-1):
                try :
                    bus_id_with_dash = bus_id[:i+1] + '-' + bus_id[i+1:]
                    bus_id = foshan_bus_dict[bus_id_with_dash]['车牌']
                    bus = foshan_bus_dict[bus_id]
                    output_str = f'车牌: {bus_id}\n'
                    for key in bus:
                        output_str += f'{key}: {bus[key]}\n'
                    output_str += f'以上信息来自佛山公交百科\n数据日期: {datadate}'
                    yield event.plain_result(output_str)
                    return
                except:
                    pass

        if city == '' or city == '江门' or city == 'JM' or city == 'jm' or city == 'J' or city == 'j':
            with open('../cha cha bus/cha-cha-bus json/Jiangmen Bus List.json', 'r', encoding='utf-8') as f:
                jiangmen_bus_dict = json.load(f)
                datadate = jiangmen_bus_dict['date']

            # 尝试找5位数+‘D’
            try :
                bus_id_with_D = bus_id + 'D'
                bus = jiangmen_bus_dict[bus_id_with_D]
                output_str = f'车牌: 粤J {bus_id_with_D}\n'
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自五邑交通维基\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                found = 1
            except:
                pass

            # 尝试找原有输入
            try :
                bus = jiangmen_bus_dict[bus_id]
                output_str = f'车牌: 粤J {bus_id}\n'
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自五邑交通维基\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                found = 1
            except:
                pass

            if found == 1 :
                return

        if city == '' or city == '东莞' or city == 'DG' or city == 'dg' or city == 'S' or city == 's':
            with open('../cha cha bus/cha-cha-bus json/Dongguan Bus List.json', 'r', encoding='utf-8') as f:
                dongguan_bus_dict = json.load(f)
                datadate = dongguan_bus_dict['date']

            # 尝试找‘粤S’+5位数+‘D’
            try :
                bus_id_with_D = '粤S'+bus_id + 'D'
                bus = dongguan_bus_dict[bus_id_with_D]
                output_str = f'车牌: {bus_id_with_D}\n'
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自东莞道路研究社\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                found = 1
            except:
                pass

            # 尝试找‘粤S’+原有输入
            try :
                _bus_id = '粤S'+bus_id
                bus = dongguan_bus_dict[_bus_id]
                output_str = f'车牌: {_bus_id}\n'
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自东莞道路研究社\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                found = 1
            except:
                pass

            # 尝试找‘场内粤S’+原有输入
            try :
                _bus_id = '场内粤S'+bus_id
                bus = dongguan_bus_dict[_bus_id]
                output_str = f'车牌: {_bus_id}\n'
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自东莞道路研究社\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                found = 1
            except:
                pass

            if found == 1 :
                return

        if city == '' or city == '中山' or city == 'ZS' or city == 'zs' or city == 'T' or city == 't':
            with open('../cha cha bus/cha-cha-bus json/Zhongshan Bus List.json', 'r', encoding='utf-8') as f:
                zhongshan_bus_dict = json.load(f)
                datadate = zhongshan_bus_dict['date']

            # 尝试找5位数+‘D’
            try :
                bus_id_with_D = bus_id + 'D'
                bus = zhongshan_bus_dict[bus_id_with_D]
                output_str = f'车牌: 粤T {bus_id_with_D}\n'
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自中山公交百科\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                found = 1
            except:
                pass

            # 尝试找原有输入
            try :
                bus = zhongshan_bus_dict[bus_id]
                output_str = f'车牌: 粤T {bus_id}\n'
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n'
                output_str += f'以上信息来自中山公交百科\n数据日期: {datadate}'
                yield event.plain_result(output_str)
                found = 1
            except:
                pass

            if found == 1 :
                return

        if found == 0 :
            yield event.plain_result(f'找不到{bus_id}的数据')


    # 票价表功能
    @filter.command("pjb", alias={'票价表', 'faretable', 'ft'})
    async def pjb(self, event: AstrMessageEvent, line: str, city: str = ''):
        """查询公交线路票价表功能""" 
        user_name = event.get_sender_name()
        message_str = event.message_str # 用户发的纯文本消息字符串
        message_chain = event.get_messages() # 用户所发的消息的消息链 # from astrbot.api.message_components import *
        logger.info(message_chain)

        found = 0
        line = line.upper()
        '''下面是指令核心部分'''

        if city == '' or city == '深圳' or city == 'SZ' or city == 'sz' or city == 'B' or city == 'b':
            image_path = f"../cha cha bus/cha-cha-bus png/shenzhen/{line}.png"
            if os.path.exists(image_path):
                yield event.chain_result([Comp.Image.fromFileSystem(image_path)])
                return
            
        if city == '' or city == '广州' or city == 'GZ' or city == 'gz' or city == 'A' or city == 'a':
            image_path = f"../cha cha bus/cha-cha-bus png/guangzhou/{line}.png"
            if os.path.exists(image_path):
                yield event.chain_result([Comp.Image.fromFileSystem(image_path)])
                return
            
        if city == '' or city == '佛山' or city == 'FS' or city == 'fs' or city == 'E' or city == 'e' or city == 'X' or city == 'x' or city == 'Y' or city == 'y':
            image_path = f"../cha cha bus/cha-cha-bus png/foshan/{line}.png"
            if os.path.exists(image_path):
                yield event.chain_result([Comp.Image.fromFileSystem(image_path)])
                return
            
        if city == '' or city == '江门' or city == 'JM' or city == 'jm' or city == 'J' or city == 'j':
            image_path = f"../cha cha bus/cha-cha-bus png/jiangmen/{line}.png"
            if os.path.exists(image_path):
                yield event.chain_result([Comp.Image.fromFileSystem(image_path)])
                return
            
        if city == '' or city == '惠州' or city == 'HZ' or city == 'hz' or city == 'L' or city == 'l':
            image_path = f"../cha cha bus/cha-cha-bus png/huizhou/{line}.png"
            if os.path.exists(image_path):
                yield event.chain_result([Comp.Image.fromFileSystem(image_path)])
                return
            
        if city == '' or city == '中山' or city == 'ZS' or city == 'zs' or city == 'T' or city == 't':
            image_path = f"../cha cha bus/cha-cha-bus png/zhongshan/{line}.png"
            if os.path.exists(image_path):
                yield event.chain_result([Comp.Image.fromFileSystem(image_path)])
                return

        if found == 0 :
            yield event.plain_result(f'找不到{line}的票价表')     

    # 档案站查询功能
    @filter.command("bp", alias={'档案站', 'daz', 'buspedia'})
    async def bp(self, event: AstrMessageEvent, bus_id: str, city: str = ''):
        """查询档案站数据功能""" 
        user_name = event.get_sender_name()
        message_str = event.message_str # 用户发的纯文本消息字符串
        message_chain = event.get_messages() # 用户所发的消息的消息链 # from astrbot.api.message_components import *
        logger.info(message_chain)

        found = 0
        bus_id = bus_id.upper()
        '''下面是指令核心部分'''

        if city == '' :
            yield event.plain_result('请给出城市名后查询\n示例：档案站 38362 深圳')
            return

        elif city != '' :
            with open('../cha cha bus/cha-cha-bus json/BP Region List.json', 'r', encoding='utf-8') as f:
                bp_region_dict = json.load(f)
            if city not in bp_region_dict:
                yield event.plain_result(f'城市输入有误')
                return
            city_id = str(bp_region_dict[city])
            yield event.plain_result('正在查询，请耐心等待10~30秒')

            try :
                process = await asyncio.create_subprocess_exec(
                    'python', '../cha cha bus/cha-cha-bus spider/Buspedia Spider.py', 
                    bus_id, city_id,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=90)
                output_data = json.loads(stdout.decode())
                title, bus_id_dict, bus_type_dict = output_data

                output_str = title + '\n---车辆信息---'
                for key in bus_id_dict:
                    output_str += f'\n{key}: {bus_id_dict[key]}'
                output_str += '\n---车型配置---'
                for key in bus_type_dict:
                    output_str += f'\n{key}: {bus_type_dict[key]}'
                yield event.plain_result(output_str)

            except :
                yield event.plain_result("好像没查到呢……\n请检查关键词后重试")
                return

    # 车牌识图功能-测试中
    @filter.command("cpsb", alias={'车牌识别', 'sbcp', '识别车牌'})
    async def cpsb(self, event: AstrMessageEvent,):
        """""" 
        message_str = event.message_str # 用户发的纯文本消息字符串
        message_chain = event.get_messages() # 用户所发的消息的消息链 # from astrbot.api.message_components import *
        logger.info(message_chain)

        '''下面是指令核心部分'''

        # 获取用户id并更新等待状态
        user_id = event.get_sender_id()
        user_name = event.get_sender_name()
        self.waiting_users[user_id] = "waiting_for_jpg"
        yield event.plain_result("请发送截图，若消息已附带则稍等")

    async def download_image_from_url(self, url: str) -> bytes:
        """从URL下载图片，返回字节数据"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    raise Exception(f"图片下载失败，HTTP状态码：{resp.status}")
                
    # 接受所有消息的逻辑 用于车牌识别功能
    @filter.event_message_type(filter.EventMessageType.ALL)

    async def cpsb_main(self, event: AstrMessageEvent):
        message_str = event.message_str # 获取消息的纯文本内容
        message_chain = event.get_messages() # 获取消息的消息链
        
        # 判断用户等待状态
        user_id = event.get_sender_id()
        if user_id in self.waiting_users and self.waiting_users[user_id] == "waiting_for_jpg":
            for msg in message_chain:
                if msg.type == 'Image':
                    # yield event.plain_result("正在识别，请稍等")

                    # 接收并保存图片文件
                    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
                    file_name = f"{current_time}_{user_id}.jpg"
                    tmp_jpg_path = os.path.join('../cha cha bus/cha-cha-bus tmp-jpg/', file_name)
                    image_url = msg.file

                    async with aiohttp.ClientSession() as session:
                        async with session.get(image_url) as response:
                            if response.status == 200:
                                image_bytes = await response.read()
                                with open(tmp_jpg_path, "wb") as f:
                                    f.write(image_bytes)
                                save_msg = f"\n图片已保存到：{tmp_jpg_path}"
                            else:
                                yield event.plain_result(f"下载图片失败，HTTP状态码：{response.status}")
                                return
                    
                    # EasyOCR识别程序
                    # reader = easyocr.Reader(lang_list=['en','ch_sim', ], gpu=True, download_enabled=True)
                    # result = reader.readtext(tmp_jpg_path, detail=1)

                    # out_str = ''
                    # for i in range(len(result)):
                    #     out_str += f"{result[i][1]} {result[i][2]:.3f}\n"
                    # yield event.plain_result(out_str)

                    # HyperLPR3识别程序
                    # print('catcher')
                    # catcher = hyp3.LicensePlateCatcher()
                    # print('imread')
                    # image = cv2.imread(tmp_jpg_path)
                    # print('result')
                    # results = catcher(image)
                    # print('output')
                    # for code, confidence, type_idx, box in results:
                    #     print(f'车牌号: {code}, 置信度: {confidence:.2f}')
                    #     yield event.plain_result(f'{code} {confidence:.3f}')

                    # PaddleOCR识别程序
                    ocr = PaddleOCR(
                        use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
                        use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
                        use_textline_orientation=False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
                        text_detection_model_name="PP-OCRv5_mobile_det",
                        text_recognition_model_name="PP-OCRv5_mobile_rec",
                        enable_mkldnn=False
                    )
                    result = ocr.predict(tmp_jpg_path)
                    bus_list = []

                    for res in result:
                        all_texts = res['rec_texts']
                        for t in all_texts:
                            if '编号' in t:
                                bus_id = t[2:]
                                city = bus_id[:2]

                                if city == '粤B':
                                    with open('../cha cha bus/cha-cha-bus json/Shenzhen Bus List.json', 'r', encoding='utf-8') as f:
                                        data = json.load(f)
                                        bus_type = data[bus_id[2:]]['表标题']
                                        line = data[bus_id[2:]]['挂牌线路']
                                        bus_list.append(f"{bus_id}: {line} {bus_type}")

                                elif city == '粤C':
                                    with open('../cha cha bus/cha-cha-bus json/Zhuhai Bus List.json', 'r', encoding='utf-8') as f:
                                        data = json.load(f)
                                        bus_type = data[bus_id[2:]]['车型']
                                        line = data[bus_id[2:]]['所属线路']
                                        bus_list.append(f"{bus_id}: {line} {bus_type}")

                                elif city == '粤S':
                                    with open('../cha cha bus/cha-cha-bus json/Dongguan Bus List.json', 'r', encoding='utf-8') as f:
                                        data = json.load(f)
                                        bus_type = data[bus_id]['车型']
                                        line = data[bus_id]['所属线路']
                                        bus_list.append(f"{bus_id}: {line} {bus_type}")

                                elif city == '粤T':
                                    with open('../cha cha bus/cha-cha-bus json/Zhongshan Bus List.json', 'r', encoding='utf-8') as f:
                                        data = json.load(f)
                                        bus_type = data[bus_id[2:]]['车型']
                                        line = data[bus_id[2:]]['所属线路']
                                        bus_list.append(f"{bus_id}: {line} {bus_type}")

                    output_str = ''
                    for bus in bus_list:
                        output_str += bus + '\n'
                    
                    if output_str != '':
                        yield event.plain_result(output_str.strip())
                    else:
                        yield event.plain_result("好像没识别到车牌，暂时只支持部分城市，且是车来了软件截图")

                    # 移除图片文件
                    os.remove(tmp_jpg_path)
                    # 移除用户等待状态
                    self.waiting_users[user_id] = None

        

    async def terminate(self):
        """可选择实现异步的插件销毁方法，当插件被卸载/停用时会调用。"""


