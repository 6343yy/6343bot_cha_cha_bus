from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from botpy.types.message import Keyboard, MarkdownPayload, Ark
from astrbot.api.message_components import Node, Plain, Image
import astrbot.api.message_components as Comp
import os
import cv2
import json
import subprocess
import asyncio
import datetime
import uuid  # 用于生成唯一文件名
import aiohttp  # 异步请求下载网络图片
import base64
from paddleocr import PaddleOCR
import openpyxl as xl
'''相对路径原点是astrbot的main.py'''

@register("6343bot_cha_cha_bus", "cha_cha_bus", "查询公交车牌、线路信息", "0.2.0")
class ChaChaBus(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.waiting_users = {} # 等待用户输入的用户列表 用于车牌识别功能

        with open('../cha cha bus/cha-cha-bus json/Ids and Cities.json', 'r', encoding='utf-8') as f:
            self.id_city_dict = json.load(f)

    async def initialize(self):
        # with open("../cha cha bus/cha-cha-bus xlsx/XiangeIdList3-16.xlsx", "r", encoding="utf-8") as f:
        self.shenzhen_bus_sheet = xl.load_workbook("../cha cha bus/cha-cha-bus xlsx/XiangeIdList.xlsx").active
        self.huizhou_bus_book = xl.load_workbook("../cha cha bus/cha-cha-bus xlsx/BgeHuizhouList.xlsx")
        self.heyuan_bus_book = xl.load_workbook("../cha cha bus/cha-cha-bus xlsx/BgeHeyuanList.xlsx")

        self.keyboard_4 = Keyboard(
            content={
                "rows": [
                    {
                        "buttons": [
                            {
                                "id": "ccp_btn",
                                "render_data": {
                                    "label": "查车牌",
                                    "style": 1
                                },
                                "action": {
                                    "type": 2,  # 回调按钮
                                    "permission": {"type": 2},
                                    "data": "/查车牌"
                                }
                            },
                            {
                                "id": "cpc_btn",
                                "render_data": {
                                    "label": "查配车",
                                    "style": 1
                                },
                                "action": {
                                    "type": 2,  # 回调按钮
                                    "permission": {"type": 2},
                                    "data": "/查配车"
                                }
                            },
                            {
                                "id": "daz_btn",
                                "render_data": {
                                    "label": "档案站",
                                    "style": 1
                                },
                                "action": {
                                    "type": 2,  # 回调按钮
                                    "permission": {"type": 2},
                                    "data": "/档案站"
                                }
                            },
                            {
                                "id": "pjb_btn",
                                "render_data": {
                                    "label": "票价表",
                                    "style": 1
                                },
                                "action": {
                                    "type": 2,  # 回调按钮
                                    "permission": {"type": 2},
                                    "data": "/票价表"
                                }
                            }
                        ]
                    }
                ]
            }
        )

    # 查车牌功能
    @filter.command("ccp", alias={'查车牌', '查车', 'ccb', 'CCB', 'CCP'})
    async def ccp(self, event: AstrMessageEvent, bus_id: str, city: str = ''):
        """查询公交车牌功能""" # 这是 handler 的描述，将会被解析方便用户了解插件内容。建议填写。
        user_name = event.get_sender_name()
        message_str = event.message_str # 用户发的纯文本消息字符串
        message_chain = event.get_messages() # 用户所发的消息的消息链 # from astrbot.api.message_components import *
        logger.info(message_chain)

        source = event.message_obj.raw_message
        if hasattr(source, 'group_openid'):
            openid = source.group_openid
        else:
            openid = source.author.user_openid
        
        found = 0
        bus_id = bus_id.upper()

        '''下面是指令核心部分'''

        if (city == '' or city == '深圳' or city == 'SZ' or city == 'sz' or city == 'B' or city == 'b') and found == 0:
            # 特定群聊：查询json文件
            if event.message_obj.group_id in ['626319593', '1F334DBE880B7FB81A23BCD858456B6B']:
                with open('../cha cha bus/cha-cha-bus json/Shenzhen Bus List.json', 'r', encoding='utf-8') as f:
                    shenzhen_bus_dict = json.load(f)
                    output_str = ''

                # 尝试找5位数+‘D’
                try :
                    bus_id_with_D = bus_id + 'D'
                    bus = shenzhen_bus_dict[bus_id_with_D]
                    for key in bus:
                        output_str += f'{key}: {bus[key]}\n' if key != '数据日期' else ''
                    datadate = bus['数据日期']
                    output_str += f'以上信息来自深圳交通百科\n数据日期: {datadate}\n'
                    found = 1
                except:
                    pass

                # 尝试找原有输入
                try :
                    bus = shenzhen_bus_dict[bus_id]
                    for key in bus:
                        output_str += f'{key}: {bus[key]}\n' if key != '数据日期' else ''
                    datadate = bus['数据日期']
                    output_str += f'以上信息来自深圳交通百科\n数据日期: {datadate}\n'
                    found = 1
                except:
                    pass
            
            # 其他情况：查找xlsx文件
            else: 
                try:
                    bus_id_with_D = bus_id + 'D'
                    for row in range(1, self.shenzhen_bus_sheet.max_row+1):
                        if self.shenzhen_bus_sheet.cell(row=row, column=4).value == bus_id or self.shenzhen_bus_sheet.cell(row=row, column=4).value == bus_id_with_D:
                            output_str = f'车牌: 粤B {self.shenzhen_bus_sheet.cell(row=row, column=4).value}\n'
                            output_str += f'车型: {self.shenzhen_bus_sheet.cell(row=row, column=3).value}\n'
                            output_str += f'公司: {self.shenzhen_bus_sheet.cell(row=row, column=1).value}\n'
                            output_str += f'线路: {self.shenzhen_bus_sheet.cell(row=row, column=2).value}\n'
                            output_str += f'涂装: {self.shenzhen_bus_sheet.cell(row=row, column=5).value}\n'
                            output_str += f'备注: {self.shenzhen_bus_sheet.cell(row=row, column=6).value}\n'
                            output_str += f'以上信息来自贤哥数据库\n'
                            if len(str(self.shenzhen_bus_sheet.cell(row=row, column=7).value)) >= 10 :
                                output_str += f'此条数据日期: {str(self.shenzhen_bus_sheet.cell(row=row, column=7).value)[0:10]}'
                            else:
                                output_str += f'此条数据日期: {self.shenzhen_bus_sheet.cell(row=row, column=7).value}'

                            found = 1
                            break

                    # 特定群里发送带反馈按钮的markdown消息
                    if found == 1 and (
                        (hasattr(source, 'group_openid') and openid in ['657ABCB111E33BFA55712FB69D123D6A', '65D143FBF1D1E9FEFFBB16385A3E2CBD']) or
                        (not hasattr(source, 'group_openid'))
                    ) :
                        markdown = MarkdownPayload(
                            content=output_str
                        )
                        keyboard_6 = Keyboard(
                            content={
                                "rows": [
                                    {
                                        "buttons": [
                                            {
                                                "id": "ccp_btn",
                                                "render_data": {
                                                    "label": "查车牌",
                                                    "style": 1
                                                },
                                                "action": {
                                                    "type": 2,  # 回调按钮
                                                    "permission": {"type": 2},
                                                    "data": "/查车牌"
                                                }
                                            },
                                            {
                                                "id": "cpc_btn",
                                                "render_data": {
                                                    "label": "查配车",
                                                    "style": 1
                                                },
                                                "action": {
                                                    "type": 2,  # 回调按钮
                                                    "permission": {"type": 2},
                                                    "data": f"/查配车"
                                                }
                                            },
                                            {
                                                "id": "daz_btn",
                                                "render_data": {
                                                    "label": "档案站",
                                                    "style": 1
                                                },
                                                "action": {
                                                    "type": 2,  # 回调按钮
                                                    "permission": {"type": 2},
                                                    "data": "/档案站"
                                                }
                                            },
                                            {
                                                "id": "pjb_btn",
                                                "render_data": {
                                                    "label": "票价表",
                                                    "style": 1
                                                },
                                                "action": {
                                                    "type": 2,  # 回调按钮
                                                    "permission": {"type": 2},
                                                    "data": "/票价表"
                                                }
                                            }
                                        ]
                                    },
                                    {
                                        "buttons": [
                                            {
                                                "id": "fk_btn",
                                                "render_data": {
                                                    "label": "反馈",
                                                    "style": 0
                                                },
                                                "action": {
                                                    "type": 2,  # 回调按钮
                                                    "permission": {"type": 2},
                                                    "data": f"/反馈 {bus_id}"
                                                }
                                            },
                                            {
                                                "id": "ckfk_btn",
                                                "render_data": {
                                                    "label": "查看反馈",
                                                    "style": 0
                                                },
                                                "action": {
                                                    "type": 2,  # 回调按钮
                                                    "permission": {"type": 2},
                                                    "data": f"/查看反馈"
                                                }
                                            }
                                        ]
                                    }
                                ]
                            }
                        )
                        if hasattr(source, 'group_openid'):
                            ret = await event.bot.api.post_group_message(
                                group_openid=openid,
                                msg_type=2,
                                markdown=markdown,
                                keyboard=keyboard_6,
                                msg_id=event.message_obj.message_id,
                            )
                        else:
                            ret = await event.post_c2c_message(
                                openid=openid,
                                msg_type=2,
                                markdown=markdown,
                                keyboard=keyboard_6,
                                msg_id=event.message_obj.message_id,
                            )
                        return
                except:
                    pass

        if (city == '' or city == '广州' or city == 'GZ' or city == 'gz' or city == 'A' or city == 'a') and found == 0:
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
                found = 1
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
                found = 1
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
                found = 1
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
                    found = 1
                except:
                    pass

            # 查广州车特有逻辑：车牌中含有*
            if '*' in bus_id:
                output_str = ''
                for a in ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']:
                    replaced_id = bus_id.replace('*', a)
                    if replaced_id in guangzhou_bus_dict:
                        line = guangzhou_bus_dict[replaced_id]['所属线路']
                        bus_type = guangzhou_bus_dict[replaced_id]['车型']
                        output_str += f'{replaced_id}: {line} {bus_type}\n'
                
                if output_str != '' :
                    found = 1

        if (city == '' or city == '珠海' or city == 'ZH' or city == 'zh' or city == 'C' or city == 'c') and found == 0:
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
                found = 1
            except:
                pass

           
        if (city == '' or city == '佛山' or city == 'FS' or city == 'fs' or city == 'E' or city == 'e' or city == 'X' or city == 'x' or city == 'Y' or city == 'y') and found == 0:
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
                found = 1
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
                found = 1
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
                found = 1
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
                found = 1
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
                found = 1
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
                found = 1
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
                    found = 1
                except:
                    pass

        if (city == '' or city == '江门' or city == 'JM' or city == 'jm' or city == 'J' or city == 'j') and found == 0:
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
                found = 1
            except:
                pass

        if (city == '' or city == '惠州' or city == 'HZ' or city == 'hz' or city == 'L' or city == 'l') and found == 0:
            for ws in self.huizhou_bus_book:
                for row in range(1, ws.max_row+1):
                    if ws.cell(row=row, column=4).value == f'粤L{bus_id}' or ws.cell(row=row, column=4).value == f'粤L{bus_id}D':
                        output_str = f'车牌: {ws.cell(row=row, column=4).value}\n'
                        for col in range(1, ws.max_column+1):
                            output_str += f'{ws.cell(row=1, column=col).value}: {ws.cell(row=row, column=col).value}\n' if col != 4 else ''
                        output_str += '以上信息来自B680数据库'
                        found = 1
                        break
                
                if found == 1:
                    break

        if (city == '' or city == '河源' or city == 'HY' or city == 'hy' or city == 'P' or city == 'p') and found == 0:
            for ws in self.heyuan_bus_book:
                for row in range(1, ws.max_row+1):
                    if ws.cell(row=row, column=4).value == f'粤P{bus_id}' or ws.cell(row=row, column=4).value == f'粤P{bus_id}D':
                        output_str = f'车牌: {ws.cell(row=row, column=4).value}\n'
                        for col in range(1, ws.max_column+1):
                            output_str += f'{ws.cell(row=1, column=col).value}: {ws.cell(row=row, column=col).value}\n' if col != 4 else ''
                        output_str += '以上信息来自B680数据库'
                        found = 1
                        break
                
                if found == 1:
                    break

        if (city == '' or city == '东莞' or city == 'DG' or city == 'dg' or city == 'S' or city == 's') and found == 0:
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
                found = 1
            except:
                pass


        if (city == '' or city == '中山' or city == 'ZS' or city == 'zs' or city == 'T' or city == 't') and found == 0:
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
                found = 1
            except:
                pass

        if (city == '' or city == '香港' or city == 'HK' or city == 'hk' or city == 'Z' or city == 'z') and found == 0:
            with open('../cha cha bus/cha-cha-bus json/Hong Kong Bus List.json', 'r', encoding='utf-8') as f:
                hongkong_bus_dict = json.load(f)

            # 尝试找原输入
            try :
                bus = hongkong_bus_dict[bus_id]
                if '標題' not in bus:
                    long_bus_id = bus['車牌']
                    bus_id = long_bus_id
                    bus = hongkong_bus_dict[bus_id]

                output_str = f'車牌: {bus_id}\n'
                datadate = bus['數據日期']
                for key in bus:
                    output_str += f'{key}: {bus[key]}\n' if key != '數據日期' else ' '
                output_str += f'以上信息來自香港巴士大典\n數據日期: {datadate}'
                found = 1
            except:
                pass

        
        if bus_id in self.id_city_dict: # 车牌开头和属地
            output_str = f'{bus_id}: {self.id_city_dict[bus_id]}'
            found = 1

        if found == 0 :
            output_str = f'找不到{bus_id}的数据'

        # 整体判断完后发送markdown消息
        markdown = MarkdownPayload(
            content=output_str.strip()
        )
        if hasattr(source, 'group_openid'):
            ret = await event.bot.api.post_group_message(
                group_openid=openid,
                msg_type=2,
                markdown=markdown,
                keyboard=self.keyboard_4,
                msg_id=event.message_obj.message_id,
            )
        else:
            ret = await event.post_c2c_message(
                openid=openid,
                msg_type=2,
                markdown=markdown,
                keyboard=self.keyboard_4,
                msg_id=event.message_obj.message_id,
            )


    # 票价表功能
    @filter.command("pjb", alias={'票价表', 'faretable', 'ft', 'PJB'})
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
            image_path = "../cha cha bus/cha-cha-bus png/shenzhen/"
            if os.path.exists(image_path+line+'.png'):
                yield event.chain_result([Comp.Image.fromFileSystem(image_path+line+'.png')])
                return
            elif os.path.exists(image_path+line+'.PNG'):
                yield event.chain_result([Comp.Image.fromFileSystem(image_path+line+'.PNG')])
                return
            
        if city == '' or city == '广州' or city == 'GZ' or city == 'gz' or city == 'A' or city == 'a':
            image_path = "../cha cha bus/cha-cha-bus png/guangzhou/"
            if os.path.exists(image_path+line+'.png'):
                yield event.chain_result([Comp.Image.fromFileSystem(image_path+line+'.png')])
                return
            elif os.path.exists(image_path+line+'.PNG'):
                yield event.chain_result([Comp.Image.fromFileSystem(image_path+line+'.PNG')])
                return
            
        if city == '' or city == '佛山' or city == 'FS' or city == 'fs' or city == 'E' or city == 'e' or city == 'X' or city == 'x' or city == 'Y' or city == 'y':
            image_path = "../cha cha bus/cha-cha-bus png/foshan/"
            if os.path.exists(image_path+line+'.png'):
                yield event.chain_result([Comp.Image.fromFileSystem(image_path+line+'.png')])
                return
            elif os.path.exists(image_path+line+'.PNG'):
                yield event.chain_result([Comp.Image.fromFileSystem(image_path+line+'.PNG')])
                return
            
        if city == '' or city == '江门' or city == 'JM' or city == 'jm' or city == 'J' or city == 'j':
            image_path = "../cha cha bus/cha-cha-bus png/jiangmen/"
            if os.path.exists(image_path+line+'.png'):
                yield event.chain_result([Comp.Image.fromFileSystem(image_path+line+'.png')])
                return
            elif os.path.exists(image_path+line+'.PNG'):
                yield event.chain_result([Comp.Image.fromFileSystem(image_path+line+'.PNG')])
                return
            
        if city == '' or city == '惠州' or city == 'HZ' or city == 'hz' or city == 'L' or city == 'l':
            image_path = "../cha cha bus/cha-cha-bus png/huizhou/"
            if os.path.exists(image_path+line+'.png'):
                yield event.chain_result([Comp.Image.fromFileSystem(image_path+line+'.png')])
                return
            elif os.path.exists(image_path+line+'.PNG'):
                yield event.chain_result([Comp.Image.fromFileSystem(image_path+line+'.PNG')])
                return
            
        if city == '' or city == '中山' or city == 'ZS' or city == 'zs' or city == 'T' or city == 't':
            image_path = "../cha cha bus/cha-cha-bus png/zhongshan/"
            if os.path.exists(image_path+line+'.png'):
                yield event.chain_result([Comp.Image.fromFileSystem(image_path+line+'.png')])
                return
            elif os.path.exists(image_path+line+'.PNG'):
                yield event.chain_result([Comp.Image.fromFileSystem(image_path+line+'.PNG')])
                return

        if found == 0 :
            yield event.plain_result(f'找不到{line}的票价表')     


    # 档案站查询功能
    @filter.command("bp", alias={'档案站', 'daz', 'buspedia', 'BP'})
    async def bp(self, event: AstrMessageEvent, bus_id: str, city: str = ''):
        """查询档案站数据功能""" 
        user_name = event.get_sender_name()
        message_str = event.message_str # 用户发的纯文本消息字符串
        message_chain = event.get_messages() # 用户所发的消息的消息链 # from astrbot.api.message_components import *
        logger.info(message_chain)
        
        source = event.message_obj.raw_message
        if hasattr(source, 'group_openid'):
            openid = source.group_openid
        else:
            openid = source.author.user_openid

        found = 0
        bus_id = bus_id.upper()
        output_str = ''
        '''下面是指令核心部分'''

        if city == '' :
            if bus_id[:2] in self.id_city_dict:
                city, bus_id = self.id_city_dict[bus_id[:2]], bus_id[2:]
                if '（' in city:
                    city = city[0:city.index('（')]
            else:
                output_str = '本功能使用示例:\n档案站 粤B38362D\n档案站 38362D 深圳' 
                

        with open('../cha cha bus/cha-cha-bus json/BP Region List.json', 'r', encoding='utf-8') as f:
            bp_region_dict = json.load(f)
        if city not in bp_region_dict:
            output_str = '本功能使用示例:\n档案站 粤B38362D\n档案站 38362D 深圳' 
            
        hint_msg_id = None
        if output_str == '':
            city_id = str(bp_region_dict[city])

            # 发送提示语
            try:
                if hasattr(source, 'group_openid'):
                    hint_ret = await event.bot.api.post_group_message(
                        group_openid=source.group_openid,
                        msg_type=0,                     # 文本消息
                        content="正在查询，一般需要10~30秒。若60秒后提示出错，请再试一次",
                        msg_id=event.message_obj.message_id,
                        msg_seq=1
                    )
                else:
                    hint_ret = await event.post_c2c_message(
                        openid=source.author.user_openid,
                        msg_type=0,
                        content="正在查询，一般需要10~30秒。若60秒后提示出错，请再试一次",
                        msg_id=event.message_obj.message_id,
                        msg_seq=1
                    )
                # 提取消息 ID
                if isinstance(hint_ret, dict):
                    hint_msg_id = hint_ret.get('id')
                else:
                    hint_msg_id = getattr(hint_ret, 'id', None)
            except Exception:
                hint_msg_id = None  # 发送失败则无法撤回，继续主流程

            # 开始查询
            try :
                process = await asyncio.create_subprocess_exec(
                    'python', '../cha cha bus/cha-cha-bus spider/Buspedia Spider.py', 
                    bus_id, city_id,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
                output_data = json.loads(stdout.decode())
                if output_data == [None]:
                    output_str = f'档案站未找到{bus_id}'
                else:
                    title, bus_id_dict, bus_type_dict = output_data

                    # output_str = title + '\n---车辆信息---'
                    # for key in bus_id_dict:
                    #     output_str += f'\n{key}: {bus_id_dict[key]}'
                    # output_str += '\n---车型配置---'
                    # for key in bus_type_dict:
                    #     output_str += f'\n{key}: {bus_type_dict[key]}'

                    output_str = f'车牌: {bus_id_dict['牌照']}\n车型: {bus_type_dict['型号']}\n```'
                    for key in bus_id_dict:
                        output_str += f'\n{key}: {bus_id_dict[key]}'
                    for key in bus_type_dict:
                        output_str += f'\n{key}: {bus_type_dict[key]}'

            except Exception as e:
                output_str = f'错误信息: {e}\n 若信息是 "Expecting value: line 1 column 1 (char 0)"，说明档案站没查到此车'

            finally:
                try:
                    driver.quit()
                except:
                    pass

        markdown = MarkdownPayload(
            content=output_str
        )
        if hasattr(source, 'group_openid'):
            ret = await event.bot.api.post_group_message(
                group_openid=openid,
                msg_type=2,
                markdown=markdown,
                keyboard=self.keyboard_4,
                msg_id=event.message_obj.message_id,
                msg_seq=2
            )
        else:
            ret = await event.post_c2c_message(
                openid=openid,
                msg_type=2,
                markdown=markdown,
                keyboard=self.keyboard_4,
                msg_id=event.message_obj.message_id,
                msg_seq=2
            )
        if hint_msg_id: # 撤回提示语
            try:
                from botpy.http import Route
                if hasattr(source, 'group_openid'):
                    route = Route(
                        "DELETE",
                        "/v2/groups/{group_openid}/messages/{message_id}",
                        group_openid=source.group_openid,
                        message_id=hint_msg_id,
                    )
                else:
                    route = Route(
                        "DELETE",
                        "/v2/users/{openid}/messages/{message_id}",
                        openid=source.author.user_openid,
                        message_id=hint_msg_id,
                    )
                await event.bot.api._http.request(route)
            except Exception as e:
                logger.warning(f"撤回提示消息失败: {e}")


    # 查询配车功能
    @filter.command("cxpc", alias={'查询配车', '查配车', '配车', 'CXPC', 'cpc', 'CPC', 'pc', 'PC',})
    async def cxpc(self, event: AstrMessageEvent, line: str, city: str = ''):
        """查询线路配车功能""" # 这是 handler 的描述，将会被解析方便用户了解插件内容。建议填写。
        user_name = event.get_sender_name()
        message_str = event.message_str # 用户发的纯文本消息字符串
        message_chain = event.get_messages() # 用户所发的消息的消息链 # from astrbot.api.message_components import *
        logger.info(message_chain)

        source = event.message_obj.raw_message
        if hasattr(source, 'group_openid'):
            openid = source.group_openid
        else:
            openid = source.author.user_openid
        
        found = 0
        sum_bus = 0
        result_dict = {}
        line = line.upper()
        line += '路' if '路' not in line and '线' not in line and '号' not in line else ''

        '''下面是指令核心部分'''

        if (city == '' or city == '深圳' or city == 'SZ' or city == 'sz' or city == 'B' or city == 'b') and found == 0:
            # 特定群聊：查询json文件
            if event.message_obj.group_id in ['626319593', '1F334DBE880B7FB81A23BCD858456B6B']: 
                with open('../cha cha bus/cha-cha-bus json/Shenzhen Bus List.json', 'r', encoding='utf-8') as f:
                    shenzhen_bus_dict = json.load(f)
                    dates = []
                    _line = line.replace('路', '')

                for key in shenzhen_bus_dict:
                    if (
                        '挂牌线路' not in shenzhen_bus_dict[key] or 
                        shenzhen_bus_dict[key]['数据日期'] == '2026-01-15' or
                        shenzhen_bus_dict[key]['退役日期'] != ''
                    ):
                        continue
                    if '/' in shenzhen_bus_dict[key]['挂牌线路'] :
                        lines = shenzhen_bus_dict[key]['挂牌线路'].split('/')
                    else:
                        lines = [shenzhen_bus_dict[key]['挂牌线路']]

                    for _l in lines:
                        if _line == _l:
                            bus_type = shenzhen_bus_dict[key]['表标题']
                            if bus_type not in result_dict:
                                result_dict[bus_type] = [key]
                            else:
                                result_dict[bus_type].append(key)
                            dates.append(shenzhen_bus_dict[key]['数据日期'])

                for key in result_dict:
                    result_dict[key] = sorted(result_dict[key])

                output_str = f'深圳{_line}配车：\n'
                for key, value in result_dict.items():
                    output_str += f'\n>**{key} ({len(value)})**\n>'
                    sum_bus += len(value)
                    for bus in value:
                        output_str += f'{bus} '
                
                if result_dict != {}:
                    found = 1
                    dates.sort()
                    output_str = output_str.replace(f'深圳{_line}配车：', f'深圳{_line}配车：({sum_bus})')
                    output_str += f'\n\n以上信息来自深圳交通百科\n数据日期: {dates[0]} 或之后'
            
            # 其他情况：查找xlsx文件
            else:
                for row in range(1, self.shenzhen_bus_sheet.max_row+1):
                    if line == self.shenzhen_bus_sheet.cell(row=row, column=2).value[:len(line)]:
                        note = self.shenzhen_bus_sheet.cell(row=row, column=6).value
                        if note == None or ('已报废' not in note and '已去' not in note and '已回' not in note and '支援' not in note):
                            bus_id = self.shenzhen_bus_sheet.cell(row=row, column=4).value 
                            bus_type = self.shenzhen_bus_sheet.cell(row=row, column=3).value 

                            if bus_type not in result_dict:
                                result_dict[bus_type] = [bus_id]
                            else:
                                result_dict[bus_type].append(bus_id)

                for key in result_dict:
                    result_dict[key] = sorted(result_dict[key])

                output_str = f'深圳{line}配车：'
                for key, value in result_dict.items():
                    output_str += f'\n{key} ({len(value)})\n>'
                    sum_bus += len(value)
                    for bus in value:
                        output_str += f'{bus} '
                    output_str += '\n'

                if output_str != f'深圳{line}配车：':
                    found = 1
                    output_str = output_str.replace(f'深圳{line}配车：', f'深圳{line}配车：({sum_bus})')
                    output_str += f'\n以上信息来自贤哥数据库'

        if (city == '' or city == '广州' or city == 'GZ' or city == 'gz' or city == 'A' or city == 'a') and found == 0:
            with open('../cha cha bus/cha-cha-bus json/Guangzhou Bus List.json', 'r', encoding='utf-8') as f:
                guangzhou_bus_dict = json.load(f)
                datadate = guangzhou_bus_dict['date']

                for key in guangzhou_bus_dict:
                    if (
                        '所属线路' not in guangzhou_bus_dict[key] or 
                        ('车辆状态' in guangzhou_bus_dict[key] and guangzhou_bus_dict[key]['车辆状态'] == '退役') or
                        ('车辆状态' in guangzhou_bus_dict[key] and guangzhou_bus_dict[key]['车辆状态'] == '已转交')
                    ) :
                        continue
                    if '/' in guangzhou_bus_dict[key]['所属线路'] :
                        lines = guangzhou_bus_dict[key]['所属线路'].split('/')
                    else:
                        lines = [guangzhou_bus_dict[key]['所属线路']]

                    for _line in lines:
                        if line == _line:
                            bus_type = guangzhou_bus_dict[key]['车型']
                            if bus_type not in result_dict:
                                result_dict[bus_type] = [key]
                            else:
                                result_dict[bus_type].append(key)

                for key in result_dict:
                    result_dict[key] = sorted(result_dict[key])

                output_str = f'广州{line}配车：'
                for key, value in result_dict.items():
                    output_str += f'\n{key} ({len(value)})\n>'
                    sum_bus += len(value)
                    for bus in value:
                        output_str += f'{bus} '
                    output_str += '\n'

                if output_str != f'广州{line}配车：':
                    found = 1
                    output_str = output_str.replace(f'广州{line}配车：', f'广州{line}配车：({sum_bus})')
                    output_str += f'\n以上信息来自广州交通维基\n数据日期: {datadate}'

        if (city == '' or city == '珠海' or city == 'ZH' or city == 'zh' or city == 'C' or city == 'c') and found == 0:
            with open('../cha cha bus/cha-cha-bus json/Zhuhai Bus List.json', 'r', encoding='utf-8') as f:
                zhuhai_bus_dict = json.load(f)
                datadate = zhuhai_bus_dict['date']

                for key in zhuhai_bus_dict:
                    if (
                        '所属线路' not in zhuhai_bus_dict[key] or 
                        ('车辆状态' in zhuhai_bus_dict[key] and zhuhai_bus_dict[key]['车辆状态'] == '退役')
                    ) :
                        continue
                    if '/' in zhuhai_bus_dict[key]['所属线路'] :
                        lines = zhuhai_bus_dict[key]['所属线路'].split('/')
                    else:
                        lines = [zhuhai_bus_dict[key]['所属线路']]

                    for _line in lines:
                        if line == _line:
                            bus_type = zhuhai_bus_dict[key]['车型']
                            if bus_type not in result_dict:
                                result_dict[bus_type] = [key]
                            else:
                                result_dict[bus_type].append(key)

                for key in result_dict:
                    result_dict[key] = sorted(result_dict[key])

                output_str = f'珠海{line}配车：'
                for key, value in result_dict.items():
                    output_str += f'\n{key} ({len(value)})\n>'
                    sum_bus += len(value)
                    for bus in value:
                        output_str += f'{bus} '
                    output_str += '\n'

                if output_str != f'珠海{line}配车：':
                    found = 1
                    output_str = output_str.replace(f'珠海{line}配车：', f'珠海{line}配车：({sum_bus})')
                    output_str += f'\n以上信息来自珠海交通维基\n数据日期: {datadate}'

        if (city == '' or city == '佛山' or city == 'FS' or city == 'fs' or city == 'E' or city == 'e' or city == 'X' or city == 'x' or city == 'Y' or city == 'y') and found == 0:
            with open('../cha cha bus/cha-cha-bus json/Foshan Bus List.json', 'r', encoding='utf-8') as f:
                foshan_bus_dict = json.load(f)
                datadate = foshan_bus_dict['date']

                for key in foshan_bus_dict:
                    if (
                        '所属线路' not in foshan_bus_dict[key] or 
                        ('车辆状态' in foshan_bus_dict[key] and foshan_bus_dict[key]['车辆状态'] == '退役')
                    ) :
                        continue
                    if '/' in foshan_bus_dict[key]['所属线路'] :
                        lines = foshan_bus_dict[key]['所属线路'].split('/')
                    else:
                        lines = [foshan_bus_dict[key]['所属线路']]

                    for _line in lines:
                        if line == _line:
                            bus_type = foshan_bus_dict[key]['车型']
                            if bus_type not in result_dict:
                                result_dict[bus_type] = [key]
                            else:
                                result_dict[bus_type].append(key)

                for key in result_dict:
                    result_dict[key] = sorted(result_dict[key])

                output_str = f'佛山{line}配车：'
                for key, value in result_dict.items():
                    output_str += f'\n{key} ({len(value)})\n>'
                    sum_bus += len(value)
                    for bus in value:
                        output_str += f'{bus} '
                    output_str += '\n'

                if output_str != f'佛山{line}配车：':
                    found = 1
                    output_str = output_str.replace(f'佛山{line}配车：', f'佛山{line}配车：({sum_bus})')
                    output_str += f'\n以上信息来自佛山公交百科\n数据日期: {datadate}'

        if (city == '' or city == '江门' or city == 'JM' or city == 'jm' or city == 'J' or city == 'j') and found == 0:
            with open('../cha cha bus/cha-cha-bus json/Jiangmen Bus List.json', 'r', encoding='utf-8') as f:
                jiangmen_bus_dict = json.load(f)
                datadate = jiangmen_bus_dict['date']

                for key in jiangmen_bus_dict:
                    if (
                        '所属线路' not in jiangmen_bus_dict[key] or 
                        ('车辆状态' in jiangmen_bus_dict[key] and jiangmen_bus_dict[key]['车辆状态'] == '退役')
                    ) :
                        continue
                    if '/' in jiangmen_bus_dict[key]['所属线路'] :
                        lines = jiangmen_bus_dict[key]['所属线路'].split('/')
                    else:
                        lines = [jiangmen_bus_dict[key]['所属线路']]

                    for _line in lines:
                        if line == _line:
                            bus_type = jiangmen_bus_dict[key]['车型']
                            if bus_type not in result_dict:
                                result_dict[bus_type] = [key]
                            else:
                                result_dict[bus_type].append(key)

                for key in result_dict:
                    result_dict[key] = sorted(result_dict[key])

                output_str = f'江门{line}配车：'
                for key, value in result_dict.items():
                    output_str += f'\n{key} ({len(value)})\n>'
                    sum_bus += len(value)
                    for bus in value:
                        output_str += f'{bus} '
                    output_str += '\n'

                if output_str != f'江门{line}配车：':
                    found = 1
                    output_str = output_str.replace(f'江门{line}配车：', f'江门{line}配车：({sum_bus})')
                    output_str += f'\n以上信息来自五邑交通维基\n数据日期: {datadate}'  

        if (city == '' or city == '惠州' or city == 'HZ' or city == 'hz' or city == 'L' or city == 'l') and found == 0:
            for ws in self.huizhou_bus_book:
                for row in range(1, ws.max_row+1):
                    if ws.cell(row=row, column=2).value != None and line == ws.cell(row=row, column=2).value[:len(line)]:
                        status = ws.cell(row=row, column=7).value
                        if status != None and ('停备' not in status and '退役' not in status and '转售' not in status):
                            bus_id = ws.cell(row=row, column=4).value 
                            bus_type = ws.cell(row=row, column=3).value 

                            if bus_type not in result_dict:
                                result_dict[bus_type] = [bus_id]
                            else:
                                result_dict[bus_type].append(bus_id)

                for key in result_dict:
                    result_dict[key] = sorted(result_dict[key])

            output_str = f'惠州{line}配车：'
            for key, value in result_dict.items():
                output_str += f'\n{key} ({len(value)})\n>'
                sum_bus += len(value)
                for bus in value:
                    output_str += f'{bus[2:]} '
                output_str += '\n'

            if output_str != f'惠州{line}配车：':
                found = 1
                output_str = output_str.replace(f'惠州{line}配车：', f'惠州{line}配车：({sum_bus})')
                output_str += f'\n以上信息来自B680数据库'

        if (city == '' or city == '河源' or city == 'HY' or city == 'hy' or city == 'P' or city == 'p') and found == 0:
            for ws in self.heyuan_bus_book:
                for row in range(1, ws.max_row+1):
                    if ws.cell(row=row, column=2).value != None and line == ws.cell(row=row, column=2).value[:len(line)]:
                        status = ws.cell(row=row, column=7).value
                        if status != None and ('停备' not in status and '退役' not in status and '转售' not in status):
                            bus_id = ws.cell(row=row, column=4).value 
                            bus_type = ws.cell(row=row, column=3).value 

                            if bus_type not in result_dict:
                                result_dict[bus_type] = [bus_id]
                            else:
                                result_dict[bus_type].append(bus_id)

            for key in result_dict:
                result_dict[key] = sorted(result_dict[key])

            output_str = f'河源{line}配车：'
            for key, value in result_dict.items():
                output_str += f'\n{key} ({len(value)})\n>'
                sum_bus += len(value)
                for bus in value:
                    output_str += f'{bus[2:]} '
                output_str += '\n'

            if output_str != f'河源{line}配车：':
                found = 1
                output_str = output_str.replace(f'河源{line}配车：', f'河源{line}配车：({sum_bus})')
                output_str += f'\n以上信息来自B680数据库'

        if (city == '' or city == '东莞' or city == 'DG' or city == 'dg' or city == 'S' or city == 's') and found == 0:
            with open('../cha cha bus/cha-cha-bus json/Dongguan Bus List.json', 'r', encoding='utf-8') as f:
                dongguan_bus_dict = json.load(f)
                datadate = dongguan_bus_dict['date']

                for key in dongguan_bus_dict:
                    if (
                        '所属线路' not in dongguan_bus_dict[key] or 
                        ('车辆状态' in dongguan_bus_dict[key] and dongguan_bus_dict[key]['车辆状态'] == '退役')
                    ) :
                        continue
                    if '/' in dongguan_bus_dict[key]['所属线路'] :
                        lines = dongguan_bus_dict[key]['所属线路'].split('/')
                    else:
                        lines = [dongguan_bus_dict[key]['所属线路']]

                    for _line in lines:
                        if line == _line:
                            bus_type = dongguan_bus_dict[key]['车型']
                            if bus_type not in result_dict:
                                result_dict[bus_type] = [key]
                            else:
                                result_dict[bus_type].append(key)

                for key in result_dict:
                    result_dict[key] = sorted(result_dict[key])

                output_str = f'东莞{line}配车：'
                for key, value in result_dict.items():
                    output_str += f'\n{key} ({len(value)})\n>'
                    sum_bus += len(value)
                    for bus in value:
                        output_str += f'{bus[2:]} ' if bus[:2] == '粤S' else f'{bus} '
                    output_str += '\n'

                if output_str != f'东莞{line}配车：':
                    found = 1
                    output_str = output_str.replace(f'东莞{line}配车：', f'东莞{line}配车：({sum_bus})')
                    output_str += f'\n以上信息来自东莞道路研究社\n数据日期: {datadate}'                    

        if (city == '' or city == '中山' or city == 'ZS' or city == 'zs' or city == 'T' or city == 't') and found == 0:
            with open('../cha cha bus/cha-cha-bus json/Zhongshan Bus List.json', 'r', encoding='utf-8') as f:
                zhongshan_bus_dict = json.load(f)
                datadate = zhongshan_bus_dict['date']

                for key in zhongshan_bus_dict:
                    if (
                        '所属线路' not in zhongshan_bus_dict[key] or 
                        ('车辆状态' in zhongshan_bus_dict[key] and zhongshan_bus_dict[key]['车辆状态'] == '退役')
                    ) :
                        continue
                    if '/' in zhongshan_bus_dict[key]['所属线路'] :
                        lines = zhongshan_bus_dict[key]['所属线路'].split('/')
                    else:
                        lines = [zhongshan_bus_dict[key]['所属线路']]

                    for _line in lines:
                        if line == _line:
                            bus_type = zhongshan_bus_dict[key]['车型']
                            if bus_type not in result_dict:
                                result_dict[bus_type] = [key]
                            else:
                                result_dict[bus_type].append(key)

                for key in result_dict:
                    result_dict[key] = sorted(result_dict[key])

                output_str = f'中山{line}配车：'
                for key, value in result_dict.items():
                    output_str += f'\n{key} ({len(value)})\n>'
                    sum_bus += len(value)
                    for bus in value:
                        output_str += f'{bus} '
                    output_str += '\n'

                if output_str != f'中山{line}配车：':
                    found = 1
                    output_str = output_str.replace(f'中山{line}配车：', f'中山{line}配车：({sum_bus})')
                    output_str += f'\n以上信息来自中山公交百科\n数据日期: {datadate}'

        if (city == '' or city == '香港' or city == 'HK' or city == 'hk' or city == 'Z' or city == 'z') and found == 0:
            with open('../cha cha bus/cha-cha-bus json/Hong Kong Bus List.json', 'r', encoding='utf-8') as f:
                hongkong_bus_dict = json.load(f)
                dates = []
                _line = line.replace('路', '')

                for key in hongkong_bus_dict:
                    if (
                        ('掛牌線' not in hongkong_bus_dict[key]) or
                        ('退役／轉售日期' in hongkong_bus_dict[key] and hongkong_bus_dict[key]['退役／轉售日期'] != '') or
                        ('退役/轉售日期' in hongkong_bus_dict[key] and hongkong_bus_dict[key]['退役/轉售日期'] != '') or
                        ('退役日期' in hongkong_bus_dict[key] and hongkong_bus_dict[key]['退役日期'] != '')
                    ) :
                        continue

                    if '/' in hongkong_bus_dict[key]['掛牌線'] :
                        lines = hongkong_bus_dict[key]['掛牌線'].split('/')
                    else:
                        lines = [hongkong_bus_dict[key]['掛牌線']]

                    for _l in lines:
                        if _line == _l:
                            bus_type = hongkong_bus_dict[key]['標題']
                            if bus_type not in result_dict:
                                result_dict[bus_type] = [key]
                            else:
                                result_dict[bus_type].append(key)
                            dates.append(hongkong_bus_dict[key]['數據日期'])

                for key in result_dict:
                    result_dict[key] = sorted(result_dict[key])

                output_str = f'香港{_line}配车：'
                for key, value in result_dict.items():
                    output_str += f'\n{key} ({len(value)})\n>'
                    sum_bus += len(value)
                    for bus in value:
                        output_str += f'{bus} '
                    output_str += '\n'

                if output_str != f'香港{_line}配车：':
                    found = 1
                    dates.sort()
                    output_str = output_str.replace(f'香港{_line}配车：', f'香港{_line}配车：({sum_bus})')
                    output_str += f'\n以上信息來自香港巴士大典\n數據日期: {dates[0]} 或之後'

        if found == 0:
            output_str = f'找不到{line}的配车'

        # 整体判断完后发送markdown消息
        markdown = MarkdownPayload(
            content=output_str.strip()
        )
        if hasattr(source, 'group_openid'):
            ret = await event.bot.api.post_group_message(
                group_openid=openid,
                msg_type=2,
                markdown=markdown,
                keyboard=self.keyboard_4,
                msg_id=event.message_obj.message_id,
            )
        else:
            ret = await event.post_c2c_message(
                openid=openid,
                msg_type=2,
                markdown=markdown,
                keyboard=self.keyboard_4,
                msg_id=event.message_obj.message_id,
            )
            

    # 重新读取表格
    @filter.command("reload", alias={'读取表格'})
    async def reload(self, event: AstrMessageEvent,):
        source = event.message_obj.raw_message
        if hasattr(source, 'group_openid'):
            return
        else:
            openid = source.author.user_openid

        self.shenzhen_bus_sheet = xl.load_workbook("../cha cha bus/cha-cha-bus xlsx/XiangeIdList.xlsx").active
        self.huizhou_bus_book = xl.load_workbook("../cha cha bus/cha-cha-bus xlsx/BgeHuizhouList.xlsx")
        self.heyuan_bus_book = xl.load_workbook("../cha cha bus/cha-cha-bus xlsx/BgeHeyuanList.xlsx")

        markdown = MarkdownPayload(
            content='读取成功'
        )

        ret = await event.post_c2c_message(
            openid=openid,
            msg_type=2,
            markdown=markdown,
            keyboard=self.keyboard_4,
            msg_id=event.message_obj.message_id,
        )


    # ark消息测试
    @filter.command("ark")
    async def ark(self, event: AstrMessageEvent):
        source = event.message_obj.raw_message

        # 仅支持私聊
        if not hasattr(source, 'author'):
            yield event.plain_result("该功能暂时只支持私聊")
            return

        openid = source.author.user_openid

        # 构造 ARK 消息
        ark = Ark(
            template_id=23,
            kv=[
                {"key": "#DESC#", "value": "机器人订阅消息"},
                {"key": "#PROMPT#", "value": "XX机器人"},
                {"key": "#LIST#", "obj": [
                    {"obj_kv": [{"key": "desc", "value": "项目A"}]},
                    {"obj_kv": [{"key": "desc", "value": "项目B"}, {"key": "link", "value": "https://q.qq.com"}]},
                    {"obj_kv": [{"key": "desc", "value": "项目C"}]}
                ]}
            ]
        )

        try:
            ret = await event.post_c2c_message(
                openid=openid,
                msg_type=3,  # ARK 消息
                ark=ark,
                msg_id=event.message_obj.message_id,
            )
            yield event.plain_result("ARK 消息发送成功！")
        except Exception as e:
            yield event.plain_result(f"发送失败: {e}")



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
        yield event.plain_result("该功能暂未启用")

    async def download_image_from_url(self, url: str) -> bytes:
        """从URL下载图片，返回字节数据"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    raise Exception(f"图片下载失败，HTTP状态码：{resp.status}")
                


    # 接受所有消息的逻辑 用于车牌识别功能
    '''@filter.event_message_type(filter.EventMessageType.ALL)

    async def cpsb_main(self, event: AstrMessageEvent):
        message_str = event.message_str # 获取消息的纯文本内容
        message_chain = event.get_messages() # 获取消息的消息链
        
        # 判断用户等待状态
        user_id = event.get_sender_id()
        # if user_id in self.waiting_users and self.waiting_users[user_id] == "waiting_for_jpg":
        for msg in message_chain:
            if msg.type == 'Image':
                # yield event.plain_result("正在识别，请稍等")

                # 接收并保存图片文件
                current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
                file_name = f"{current_time}_{user_id}.jpg"
                tmp_jpg_path = os.path.join('../cha cha bus/cha-cha-bus tmp-jpg/', file_name)
                image_url = msg.file

                # 下载图片
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url) as response:
                        if response.status == 200:
                            image_bytes = await response.read()
                            with open(tmp_jpg_path, "wb") as f:
                                f.write(image_bytes)
                        else:
                            yield event.plain_result(f"下载图片失败，HTTP状态码：{response.status}")
                            return
                
                # EasyOCR识别程序
                # reader = easyocr.Reader(lang_list=['en','ch_sim', ], gpu=True, download_enabled=True)
                # result = reader.readtext(tmp_jpg_path, detail=1)

                # output_str = ''
                # for i in range(len(result)):
                #     output_str += f"{result[i][1]} {result[i][2]:.3f}\n"
                # yield event.plain_result(output_str)

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
                # self.waiting_users[user_id] = None
    '''
        
        
    async def terminate(self):
        """可选择实现异步的插件销毁方法，当插件被卸载/停用时会调用。"""
