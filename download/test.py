#encoding: utf8

import random 
import execjs  
import pdb
import requests
import json

'''
测试 vjkl5 代码用的
'''
def do_test():
    vjkl5='34d09dff9cd2de016018d30aaa94d63c5950344b'
    vl5x='5e948c23b00d13d283015caa'
    s = ctx.call('getKey', vjkl5)

    if s == vl5x:
        print "decocde matched."
    else:
        print "Error in decode"

    print "GUID:", guid()


def guid_part(): 
    return hex(int((random.random() + 1) * 0x10000))[3:] 

def guid():
    uf = guid_part
    r = map(lambda x:uf(), range(8))
    uid = '%s%s-%s-%s%s-%s%s%s' % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7])

    return uid

def get_list(params):
    #民事列表url, 获取vjkl5
    civil_law_url = 'http://wenshu.court.gov.cn/List/List?sorttype=1&conditions=searchWord+2+AJLX++%E6%A1%88%E4%BB%B6%E7%B1%BB%E5%9E%8B:%E6%B0%91%E4%BA%8B%E6%A1%88%E4%BB%B6'

    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36', 
            'Origin': 'http://wenshu.court.gov.cn',
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': civil_law_url
    }


    proxies = {
            "http":  "adslexp01.web.zw.ted:8100",
            "https": "adslexp01.web.zw.ted:8100",
    }

    proxies = {
            "http":  "adslexp01.web.zw.ted:8099",
            "https": "adslexp01.web.zw.ted:8099",
    }

    proxies = {
            "http":  "adslspider17.shop.zw.ted:8080",
            "https": "adslspider17.shop.zw.ted:8080",
    }

    proxies = {
            "http": "adslspider16.shop.zw.vm.ted:8080",
            "https": "adslspider16.shop.zw.vm.ted:8080",
    }

    proxies = {
            "http": "adslspider01.shop.zw.vm.ted:8080",
            "https": "adslspider01.shop.zw.vm.ted:8080",
    }

    proxies = {
            "http":  "adslproxy01.shida.sjs.ted:90",
            "https": "adslproxy01.shida.sjs.ted:90",
    }

    proxies = {
            "http": "adslspider48.web.zw.ted:9090",
            "https": "adslspider48.web.zw.ted:9090",
    }

    s = requests.session()
    s.headers.update(headers)
    r = s.get(civil_law_url, proxies=proxies)
    
    #读取cookie里的 vjkl5
    vjkl5 = r.cookies['vjkl5']

    #获取number的url, 获取number
    code_url = 'http://wenshu.court.gov.cn/ValiCode/GetCode'
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36', 
            'Origin': 'http://wenshu.court.gov.cn',
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': civil_law_url
            }

    _guid = guid()
    dg = {'guid': _guid}

    s = requests.session()
    s.headers.update(headers)
    cookies = {'vjkl5':vjkl5}
    r = s.post(code_url, data=dg, cookies=cookies, proxies=proxies)
    number = r.text

    #生成加密 vl5x
    vl5x = ctx.call('getKey', vjkl5)

    #列表url. 获取查询结果
    list_url = 'http://wenshu.court.gov.cn/List/ListContent'
    p = dict(params)
    p['vl5x'] = vl5x
    p['guid'] = _guid
    p['number'] = number

    s = requests.session()
    s.headers.update(headers)
    cookies = {'vjkl5':vjkl5}
    r = s.post(list_url, data=p, cookies=cookies, proxies=proxies)

    jstr = r.text.encode('utf8').strip().strip('"').decode('string_escape')

    return jstr

def get_all_list(param):
    cur_page_id = 1
    page_size = 20
    while True:
        jstr = get_list(param) 
        if jstr == None or len(jstr) < 100:
            print "ERROR in GET"
            continue
    
        jobj = json.loads(jstr)

        for oi in jobj[1:]:
            print oi[u'文书ID'],'\t', oi[u'案号'], '\t', oi[u'裁判日期']

        count = int(jobj[0][u'Count'])
        total_page = (count + page_size - 1) / page_size 

        if cur_page_id >= total_page:
            break
    
        cur_page_id += 1
        param['Index'] = cur_page_id


#执行本地的js  
f = open("lutil.js", 'r')  
content = f.read()  
ctx = execjs.compile(content.decode('utf8'))  

param = {"Param": "案件类型:民事案件,关键词:合同", 
         "Index": 1,
        "Page": 20,
        "Order": "法院层级",
        "Direction": "asc"}

param = {'Param': '案件类型:民事案件,法院名称:北京市昌平区人民法院,裁判日期:2018-04-02   TO 2018-04-10', 
         'Index': 1,
        'Page': 20}

get_all_list(param)
