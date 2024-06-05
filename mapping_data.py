reserved_addr = {
    '01:00:5E:00:00:05':'OSPF',
    '01:00:5E:00:00:06':'DR',
    '01:00:5E:00:00:09':'RIP',
    '01:00:5E:00:00:12':'VRRP',
    '01:00:5E:00:00:6B':'PTP',
    '01:00:5E:00:00:FB':'mDNS',
    '01:00:5E:00:00:FC':'LLMNR',
    '01:00:5E:00:01:01':'NTP',
    '01:00:5E:00:01:81':'PTP',
    '01:00:5E:00:01:82':'PTP',
    '01:00:5E:00:01:83':'PTP',
    '01:00:5E:00:01:84':'PTP',
    '01:00:5E:7F:FF:FA':'SSDP, SLPv2, and other Discovery protocol address',
    '01:00:5E:7F:FF:FB':'SSDP, SLPv2, and other Discovery protocol address',
    '01:00:5E:7F:FF:FC':'SSDP, SLPv2, and other Discovery protocol address',
    '01:00:5E:7F:FF:FD':'SSDP, SLPv2, and other Discovery protocol address',
    '01:00:5E:00:00:E6':'Reserved for Dante AV',
    '01:00:5E:00:00:E7':'Reserved for Dante AV',
    '01:00:5E:00:00:E8':'Reserved for Dante AV',
    '01:00:5E:00:00:E9':'Reserved for Dante AV',
    '91:E0:F0:01:00:00':'IEEE1722'
}

device_dict = {
    'd0:52:a8:00:67:5e':'Smart Things',
    '44:65:0d:56:cc:d3':'Amazon Echo',
    '70:ee:50:18:34:43':'Netatmo Welcome',
    'f4:f2:6d:93:51:f1':'TP-Link Day Night Cloud camera',
    '00:16:6c:ab:6b:88':'Samsung SmartCam',
    '30:8c:fb:2f:e4:b2':'Dropcam',
    '00:62:6e:51:27:2e':'Insteon Camera',
    'e8:ab:fa:19:de:4f':'Insteon Camera',
    '00:24:e4:11:18:a8':'Withings Smart Baby Monitor',
    'ec:1a:59:79:f4:89':'Belkin Wemo switch',
    '50:c7:bf:00:56:39':'TP-Link Smart plug',
    '74:c6:3b:29:d7:1d':'iHome',
    'ec:1a:59:83:28:11':'Belkin wemo motion sensor',
    '18:b4:30:25:be:e4':'NEST Protect smoke alarm',
    '70:ee:50:03:b8:ac':'Netatmo weather station',
    '00:24:e4:1b:6f:96':'Withings Smart scale',
    '74:6a:89:00:2e:25':'Blipcare Blood Pressure meter',
    '00:24:e4:20:28:c6':'Withings Aura smart sleep sensor',
    'd0:73:d5:01:83:08':'LiFX Smart Bulb',
    '18:b7:9e:02:20:44':'Triby Speaker',
    'e0:76:d0:33:bb:85':'PIX-STAR Photo-frame',
    '70:5a:0f:e4:9b:c0':'HP Printer',
}

additional_device = {
    '14:cc:20:51:33:ea':'PLink Router Bridge LAN',
    '74:2f:68:81:69:42':'Labtop',
    'b4:ce:f6:a7:a3:c2':'Android Phone',
    'd0:a6:37:df:a1:e1':'Iphone',
}

alter_device = {
    'Triby Speaker':
        ['Amazon Echo Dot', 'Sonos One'],
    
    'TP-Link WiFi Router':
        ['Philips Hue Bridge', 'TP-Link TL-WR940N', 'Gigamon Network Tap G-TAP A-TX',
        'Cisco Catalyst 3850 24 switch', 'Netgear Unmanaged Switch GS308', 'Asus router RT-N12',
        'Arlo Base station', 'Eufy Homebase 2', 'Fibaro Home Center Lite', 'SmartThings Smart Hub'],
    
    'Dropcam':
        ['Google Nest Cam Indoor', 'Blink For Home', 'HD Indoor/Outdoor IP Dome Camera HD838',
        'HD Indoor/Outdoor Mini IP Bullet Camera HD438', 'Arlo Q Camera, Amcrest 2K Camera,'
        'Netatmo Indoor Camera', 'Tapo C200 Camera', 'Yi Indoor Camera (Yi Home Camera)', 'Wyze V3 Camera',
        'TP-Link Day Night Cloud camera(NC220)', 'Insteon Camera'],
    
    'Withings Smart Baby Monitor':
        ['Google Nest Cam Indoor', 'Blink For Home', 'HD Indoor/Outdoor IP Dome Camera HD838',
        'HD Indoor/Outdoor Mini IP Bullet Camera HD438', 'Arlo Q Camera, Amcrest 2K Camera,'
        'Netatmo Indoor Camera', 'Tapo C200 Camera', 'Yi Indoor Camera (Yi Home Camera)', 'Wyze V3 Camera',
        'TP-Link Day Night Cloud camera(NC220)', 'Insteon Camera'],
    
    'Samsung SmartCam':
        ['Google Nest Cam Indoor', 'Blink For Home', 'HD Indoor/Outdoor IP Dome Camera HD838',
        'HD Indoor/Outdoor Mini IP Bullet Camera HD438', 'Arlo Q Camera, Amcrest 2K Camera,'
        'Netatmo Indoor Camera', 'Tapo C200 Camera', 'Yi Indoor Camera (Yi Home Camera)', 'Wyze V3 Camera',
        'TP-Link Day Night Cloud camera(NC220)', 'Insteon Camera'],
    
    'Belkin wemo motion sensor':
        ['Belkin Wemo switch', 'Belkin Wemo (Insight) switch'],
    
    'PIX-STAR Photo-frame':
        ['HP Printer'],
    
    'TP-Link Smart plug':
        ['Belkin Wemo Smart Plug WSP080 v1.2', 'Belkin Wemo Insight Smart Plug',
        'Konke Smart Plug K', 'Qubo Smart Plug 10A', 'Eques elf smart plug'],
    
    'Withings Smart scale':
        ['Blipcare Blood Pressure meter'],
    
    'iHome':
        ['Belkin Wemo switch', 'Belkin Wemo (Insight) switch', 'Belkin Wemo switch'],
        
    'Withings Aura smart sleep sensor':
        ['Blipcare Blood Pressure meter']
}