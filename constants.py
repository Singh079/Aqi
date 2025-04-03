"""
Constants for the AQI Monitoring Application
"""

# AQI Level Categories
AQI_LEVELS = {
    "Good": {"range": (0, 50), "color": "#55A84F", "description": "Air quality is satisfactory, and air pollution poses little or no risk."},
    "Satisfactory": {"range": (51, 100), "color": "#A3C853", "description": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."},
    "Moderate": {"range": (101, 200), "color": "#FFF833", "description": "Members of sensitive groups may experience health effects. The general public is less likely to be affected."},
    "Poor": {"range": (201, 300), "color": "#F29C33", "description": "Health alert: The risk of health effects is increased for everyone."},
    "Very Poor": {"range": (301, 400), "color": "#E93F33", "description": "Health warning of emergency conditions: everyone is more likely to be affected."},
    "Severe": {"range": (401, 500), "color": "#AF2D24", "description": "Health alert: everyone may experience more serious health effects."}
}

# Major Indian Cities with their latitude, longitude and state information
MAJOR_CITIES = {
    # Andhra Pradesh
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh"},
    "Vijayawada": {"lat": 16.5062, "lon": 80.6480, "state": "Andhra Pradesh"},
    "Guntur": {"lat": 16.3067, "lon": 80.4365, "state": "Andhra Pradesh"},
    "Nellore": {"lat": 14.4426, "lon": 79.9865, "state": "Andhra Pradesh"},
    "Kurnool": {"lat": 15.8281, "lon": 78.0373, "state": "Andhra Pradesh"},
    "Rajahmundry": {"lat": 16.9891, "lon": 81.7840, "state": "Andhra Pradesh"},
    "Tirupati": {"lat": 13.6288, "lon": 79.4192, "state": "Andhra Pradesh"},
    "Kakinada": {"lat": 16.9891, "lon": 82.2475, "state": "Andhra Pradesh"},
    "Anantapur": {"lat": 14.6819, "lon": 77.6006, "state": "Andhra Pradesh"},
    "Kadapa": {"lat": 14.4673, "lon": 78.8242, "state": "Andhra Pradesh"},
    "Eluru": {"lat": 16.7107, "lon": 81.0952, "state": "Andhra Pradesh"},
    
    # Arunachal Pradesh
    "Itanagar": {"lat": 27.0844, "lon": 93.6053, "state": "Arunachal Pradesh"},
    "Naharlagun": {"lat": 27.1044, "lon": 93.6963, "state": "Arunachal Pradesh"},
    "Pasighat": {"lat": 28.0654, "lon": 95.3280, "state": "Arunachal Pradesh"},
    "Tawang": {"lat": 27.5859, "lon": 91.8083, "state": "Arunachal Pradesh"},
    "Bomdila": {"lat": 27.2650, "lon": 92.4099, "state": "Arunachal Pradesh"},
    
    # Assam
    "Guwahati": {"lat": 26.1445, "lon": 91.7362, "state": "Assam"},
    "Silchar": {"lat": 24.8333, "lon": 92.7789, "state": "Assam"},
    "Dibrugarh": {"lat": 27.4728, "lon": 94.9120, "state": "Assam"},
    "Jorhat": {"lat": 26.7509, "lon": 94.2037, "state": "Assam"},
    "Nagaon": {"lat": 26.3500, "lon": 92.6833, "state": "Assam"},
    "Tinsukia": {"lat": 27.4924, "lon": 95.3571, "state": "Assam"},
    "Tezpur": {"lat": 26.6528, "lon": 92.7926, "state": "Assam"},
    "Bongaigaon": {"lat": 26.4784, "lon": 90.5583, "state": "Assam"},
    
    # Bihar
    "Patna": {"lat": 25.5941, "lon": 85.1376, "state": "Bihar"},
    "Gaya": {"lat": 24.7914, "lon": 84.9994, "state": "Bihar"},
    "Muzaffarpur": {"lat": 26.1197, "lon": 85.3910, "state": "Bihar"},
    "Bhagalpur": {"lat": 25.2425, "lon": 86.9842, "state": "Bihar"},
    "Darbhanga": {"lat": 26.1542, "lon": 85.8918, "state": "Bihar"},
    "Purnia": {"lat": 25.7771, "lon": 87.4753, "state": "Bihar"},
    "Arrah": {"lat": 25.5563, "lon": 84.6602, "state": "Bihar"},
    "Katihar": {"lat": 25.5541, "lon": 87.5750, "state": "Bihar"},
    "Munger": {"lat": 25.3746, "lon": 86.4738, "state": "Bihar"},
    "Chhapra": {"lat": 25.7815, "lon": 84.7503, "state": "Bihar"},
    "Bettiah": {"lat": 26.8024, "lon": 84.5101, "state": "Bihar"},
    "Saharsa": {"lat": 25.8818, "lon": 86.6005, "state": "Bihar"},
    
    # Chhattisgarh
    "Raipur": {"lat": 21.2514, "lon": 81.6296, "state": "Chhattisgarh"},
    "Bhilai": {"lat": 21.2060, "lon": 81.4283, "state": "Chhattisgarh"},
    "Bilaspur": {"lat": 22.0797, "lon": 82.1409, "state": "Chhattisgarh"},
    "Korba": {"lat": 22.3595, "lon": 82.7501, "state": "Chhattisgarh"},
    "Durg": {"lat": 21.1906, "lon": 81.2764, "state": "Chhattisgarh"},
    "Rajnandgaon": {"lat": 21.0972, "lon": 81.0370, "state": "Chhattisgarh"},
    "Jagdalpur": {"lat": 19.0723, "lon": 82.0346, "state": "Chhattisgarh"},
    "Raigarh": {"lat": 21.9059, "lon": 83.3961, "state": "Chhattisgarh"},
    "Ambikapur": {"lat": 23.1238, "lon": 83.1977, "state": "Chhattisgarh"},
    
    # Delhi
    "Delhi": {"lat": 28.7041, "lon": 77.1025, "state": "Delhi"},
    "New Delhi": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi"},
    "South Delhi": {"lat": 28.5506, "lon": 77.2203, "state": "Delhi"},
    "West Delhi": {"lat": 28.6663, "lon": 77.0675, "state": "Delhi"},
    "East Delhi": {"lat": 28.6651, "lon": 77.2856, "state": "Delhi"},
    "North Delhi": {"lat": 28.7213, "lon": 77.1908, "state": "Delhi"},
    
    # Goa
    "Panaji": {"lat": 15.4909, "lon": 73.8278, "state": "Goa"},
    "Margao": {"lat": 15.2832, "lon": 73.9862, "state": "Goa"},
    "Vasco da Gama": {"lat": 15.3981, "lon": 73.8158, "state": "Goa"},
    "Mapusa": {"lat": 15.5937, "lon": 73.8142, "state": "Goa"},
    "Ponda": {"lat": 15.4027, "lon": 74.0078, "state": "Goa"},
    
    # Gujarat
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat"},
    "Vadodara": {"lat": 22.3072, "lon": 73.1812, "state": "Gujarat"},
    "Surat": {"lat": 21.1702, "lon": 72.8311, "state": "Gujarat"},
    "Rajkot": {"lat": 22.3039, "lon": 70.8022, "state": "Gujarat"},
    "Gandhinagar": {"lat": 23.2156, "lon": 72.6369, "state": "Gujarat"},
    "Bhavnagar": {"lat": 21.7645, "lon": 72.1519, "state": "Gujarat"},
    "Jamnagar": {"lat": 22.4707, "lon": 70.0577, "state": "Gujarat"},
    "Junagadh": {"lat": 21.5222, "lon": 70.4579, "state": "Gujarat"},
    "Anand": {"lat": 22.5645, "lon": 72.9289, "state": "Gujarat"},
    "Gandhidham": {"lat": 23.0787, "lon": 70.1336, "state": "Gujarat"},
    "Nadiad": {"lat": 22.6916, "lon": 72.8634, "state": "Gujarat"},
    "Morbi": {"lat": 22.8252, "lon": 70.8407, "state": "Gujarat"},
    "Surendranagar": {"lat": 22.7255, "lon": 71.6602, "state": "Gujarat"},
    "Porbandar": {"lat": 21.6425, "lon": 69.6323, "state": "Gujarat"},
    
    # Haryana
    "Faridabad": {"lat": 28.4089, "lon": 77.3178, "state": "Haryana"},
    "Gurugram": {"lat": 28.4595, "lon": 77.0266, "state": "Haryana"},
    "Panipat": {"lat": 29.3909, "lon": 76.9635, "state": "Haryana"},
    "Ambala": {"lat": 30.3782, "lon": 76.7767, "state": "Haryana"},
    "Yamunanagar": {"lat": 30.1290, "lon": 77.2674, "state": "Haryana"},
    "Rohtak": {"lat": 28.8955, "lon": 76.6066, "state": "Haryana"},
    "Hisar": {"lat": 29.1492, "lon": 75.7217, "state": "Haryana"},
    "Karnal": {"lat": 29.6857, "lon": 76.9905, "state": "Haryana"},
    "Sonipat": {"lat": 28.9931, "lon": 77.0151, "state": "Haryana"},
    "Panchkula": {"lat": 30.6942, "lon": 76.8606, "state": "Haryana"},
    "Bhiwani": {"lat": 28.7975, "lon": 76.1322, "state": "Haryana"},
    "Sirsa": {"lat": 29.5321, "lon": 75.0318, "state": "Haryana"},
    
    # Himachal Pradesh
    "Shimla": {"lat": 31.1048, "lon": 77.1734, "state": "Himachal Pradesh"},
    "Dharamshala": {"lat": 32.2190, "lon": 76.3234, "state": "Himachal Pradesh"},
    "Solan": {"lat": 30.9045, "lon": 77.0967, "state": "Himachal Pradesh"},
    "Mandi": {"lat": 31.7080, "lon": 76.9318, "state": "Himachal Pradesh"},
    "Kullu": {"lat": 31.9592, "lon": 77.1089, "state": "Himachal Pradesh"},
    "Hamirpur": {"lat": 31.6861, "lon": 76.5212, "state": "Himachal Pradesh"},
    "Una": {"lat": 31.4685, "lon": 76.2708, "state": "Himachal Pradesh"},
    
    # Jammu & Kashmir
    "Srinagar": {"lat": 34.0837, "lon": 74.7973, "state": "Jammu & Kashmir"},
    "Jammu": {"lat": 32.7266, "lon": 74.8570, "state": "Jammu & Kashmir"},
    "Anantnag": {"lat": 33.7311, "lon": 75.1487, "state": "Jammu & Kashmir"},
    "Baramulla": {"lat": 34.2095, "lon": 74.3436, "state": "Jammu & Kashmir"},
    "Udhampur": {"lat": 32.9156, "lon": 75.1417, "state": "Jammu & Kashmir"},
    "Kathua": {"lat": 32.5834, "lon": 75.5082, "state": "Jammu & Kashmir"},
    
    # Jharkhand
    "Ranchi": {"lat": 23.3441, "lon": 85.3096, "state": "Jharkhand"},
    "Jamshedpur": {"lat": 22.8046, "lon": 86.2029, "state": "Jharkhand"},
    "Dhanbad": {"lat": 23.7957, "lon": 86.4304, "state": "Jharkhand"},
    "Bokaro": {"lat": 23.6693, "lon": 86.1511, "state": "Jharkhand"},
    "Hazaribagh": {"lat": 23.9924, "lon": 85.3637, "state": "Jharkhand"},
    "Deoghar": {"lat": 24.4800, "lon": 86.7044, "state": "Jharkhand"},
    "Giridih": {"lat": 24.1914, "lon": 86.3120, "state": "Jharkhand"},
    "Ramgarh": {"lat": 23.6314, "lon": 85.5121, "state": "Jharkhand"},
    
    # Karnataka
    "Bangalore": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
    "Mysore": {"lat": 12.2958, "lon": 76.6394, "state": "Karnataka"},
    "Hubli-Dharwad": {"lat": 15.3647, "lon": 75.1240, "state": "Karnataka"},
    "Mangalore": {"lat": 12.9141, "lon": 74.8560, "state": "Karnataka"},
    "Belgaum": {"lat": 15.8497, "lon": 74.4977, "state": "Karnataka"},
    "Davangere": {"lat": 14.4644, "lon": 75.9218, "state": "Karnataka"},
    "Bellary": {"lat": 15.1425, "lon": 76.9209, "state": "Karnataka"},
    "Gulbarga": {"lat": 17.3297, "lon": 76.8343, "state": "Karnataka"},
    "Shimoga": {"lat": 13.9299, "lon": 75.5681, "state": "Karnataka"},
    "Tumkur": {"lat": 13.3379, "lon": 77.1173, "state": "Karnataka"},
    "Bijapur": {"lat": 16.8302, "lon": 75.7100, "state": "Karnataka"},
    "Udupi": {"lat": 13.3409, "lon": 74.7421, "state": "Karnataka"},
    "Hassan": {"lat": 13.0002, "lon": 76.1003, "state": "Karnataka"},
    
    # Kerala
    "Thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366, "state": "Kerala"},
    "Kochi": {"lat": 9.9312, "lon": 76.2673, "state": "Kerala"},
    "Kozhikode": {"lat": 11.2588, "lon": 75.7804, "state": "Kerala"},
    "Thrissur": {"lat": 10.5276, "lon": 76.2144, "state": "Kerala"},
    "Kollam": {"lat": 8.8932, "lon": 76.6141, "state": "Kerala"},
    "Kannur": {"lat": 11.8745, "lon": 75.3704, "state": "Kerala"},
    "Alappuzha": {"lat": 9.4981, "lon": 76.3388, "state": "Kerala"},
    "Palakkad": {"lat": 10.7867, "lon": 76.6548, "state": "Kerala"},
    "Kottayam": {"lat": 9.5916, "lon": 76.5222, "state": "Kerala"},
    "Malappuram": {"lat": 11.0730, "lon": 76.0740, "state": "Kerala"},
    "Kasaragod": {"lat": 12.4996, "lon": 74.9869, "state": "Kerala"},
    "Pathanamthitta": {"lat": 9.2648, "lon": 76.7870, "state": "Kerala"},
    
    # Madhya Pradesh
    "Indore": {"lat": 22.7196, "lon": 75.8577, "state": "Madhya Pradesh"},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh"},
    "Jabalpur": {"lat": 23.1815, "lon": 79.9864, "state": "Madhya Pradesh"},
    "Gwalior": {"lat": 26.2183, "lon": 78.1828, "state": "Madhya Pradesh"},
    "Ujjain": {"lat": 23.1765, "lon": 75.7885, "state": "Madhya Pradesh"},
    "Sagar": {"lat": 23.8388, "lon": 78.7378, "state": "Madhya Pradesh"},
    "Dewas": {"lat": 22.9676, "lon": 76.0534, "state": "Madhya Pradesh"},
    "Satna": {"lat": 24.6005, "lon": 80.8322, "state": "Madhya Pradesh"},
    "Ratlam": {"lat": 23.3315, "lon": 75.0367, "state": "Madhya Pradesh"},
    "Rewa": {"lat": 24.5362, "lon": 81.3037, "state": "Madhya Pradesh"},
    "Katni": {"lat": 23.8343, "lon": 80.3894, "state": "Madhya Pradesh"},
    "Singrauli": {"lat": 24.1991, "lon": 82.6740, "state": "Madhya Pradesh"},
    "Burhanpur": {"lat": 21.3145, "lon": 76.2310, "state": "Madhya Pradesh"},
    
    # Maharashtra
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
    "Pune": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra"},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882, "state": "Maharashtra"},
    "Thane": {"lat": 19.2183, "lon": 72.9781, "state": "Maharashtra"},
    "Nashik": {"lat": 19.9975, "lon": 73.7898, "state": "Maharashtra"},
    "Aurangabad": {"lat": 19.8762, "lon": 75.3433, "state": "Maharashtra"},
    "Solapur": {"lat": 17.6599, "lon": 75.9064, "state": "Maharashtra"},
    "Navi Mumbai": {"lat": 19.0330, "lon": 73.0297, "state": "Maharashtra"},
    "Kalyan-Dombivli": {"lat": 19.2403, "lon": 73.1305, "state": "Maharashtra"},
    "Vasai-Virar": {"lat": 19.3919, "lon": 72.8397, "state": "Maharashtra"},
    "Kolhapur": {"lat": 16.7050, "lon": 74.2433, "state": "Maharashtra"},
    "Sangli": {"lat": 16.8524, "lon": 74.5815, "state": "Maharashtra"},
    "Malegaon": {"lat": 20.5579, "lon": 74.5089, "state": "Maharashtra"},
    "Jalgaon": {"lat": 21.0077, "lon": 75.5626, "state": "Maharashtra"},
    "Akola": {"lat": 20.7002, "lon": 77.0082, "state": "Maharashtra"},
    "Latur": {"lat": 18.4088, "lon": 76.5604, "state": "Maharashtra"},
    "Dhule": {"lat": 20.9042, "lon": 74.7749, "state": "Maharashtra"},
    "Amravati": {"lat": 20.9320, "lon": 77.7523, "state": "Maharashtra"},
    "Nanded": {"lat": 19.1383, "lon": 77.3210, "state": "Maharashtra"},
    "Kolhapur": {"lat": 16.7050, "lon": 74.2433, "state": "Maharashtra"},
    
    # Manipur
    "Imphal": {"lat": 24.8170, "lon": 93.9368, "state": "Manipur"},
    "Thoubal": {"lat": 24.6422, "lon": 94.0148, "state": "Manipur"},
    "Bishnupur": {"lat": 24.6367, "lon": 93.7661, "state": "Manipur"},
    "Churachandpur": {"lat": 24.3300, "lon": 93.6800, "state": "Manipur"},
    
    # Meghalaya
    "Shillong": {"lat": 25.5788, "lon": 91.8933, "state": "Meghalaya"},
    "Tura": {"lat": 25.5172, "lon": 90.2137, "state": "Meghalaya"},
    "Jowai": {"lat": 25.4489, "lon": 92.2029, "state": "Meghalaya"},
    "Nongstoin": {"lat": 25.5200, "lon": 91.2700, "state": "Meghalaya"},
    
    # Mizoram
    "Aizawl": {"lat": 23.7271, "lon": 92.7176, "state": "Mizoram"},
    "Lunglei": {"lat": 22.8671, "lon": 92.7655, "state": "Mizoram"},
    "Champhai": {"lat": 23.4500, "lon": 93.3300, "state": "Mizoram"},
    
    # Nagaland
    "Kohima": {"lat": 25.6751, "lon": 94.1086, "state": "Nagaland"},
    "Dimapur": {"lat": 25.9091, "lon": 93.7277, "state": "Nagaland"},
    "Mokokchung": {"lat": 26.3233, "lon": 94.5112, "state": "Nagaland"},
    "Tuensang": {"lat": 26.2356, "lon": 94.8253, "state": "Nagaland"},
    
    # Odisha
    "Bhubaneswar": {"lat": 20.2961, "lon": 85.8245, "state": "Odisha"},
    "Cuttack": {"lat": 20.4625, "lon": 85.8830, "state": "Odisha"},
    "Rourkela": {"lat": 22.2604, "lon": 84.8536, "state": "Odisha"},
    "Berhampur": {"lat": 19.3149, "lon": 84.7941, "state": "Odisha"},
    "Sambalpur": {"lat": 21.4669, "lon": 83.9756, "state": "Odisha"},
    "Puri": {"lat": 19.8106, "lon": 85.8314, "state": "Odisha"},
    "Balasore": {"lat": 21.4927, "lon": 86.9439, "state": "Odisha"},
    "Bhadrak": {"lat": 21.0571, "lon": 86.4860, "state": "Odisha"},
    "Baripada": {"lat": 21.9373, "lon": 86.7213, "state": "Odisha"},
    
    # Punjab
    "Ludhiana": {"lat": 30.9010, "lon": 75.8573, "state": "Punjab"},
    "Amritsar": {"lat": 31.6340, "lon": 74.8723, "state": "Punjab"},
    "Jalandhar": {"lat": 31.3260, "lon": 75.5762, "state": "Punjab"},
    "Patiala": {"lat": 30.3398, "lon": 76.3869, "state": "Punjab"},
    "Bathinda": {"lat": 30.2110, "lon": 74.9455, "state": "Punjab"},
    "Hoshiarpur": {"lat": 31.5143, "lon": 75.9115, "state": "Punjab"},
    "Mohali": {"lat": 30.7046, "lon": 76.7179, "state": "Punjab"},
    "Pathankot": {"lat": 32.2643, "lon": 75.6500, "state": "Punjab"},
    "Moga": {"lat": 30.8165, "lon": 75.1667, "state": "Punjab"},
    "Firozpur": {"lat": 30.9331, "lon": 74.6125, "state": "Punjab"},
    
    # Rajasthan
    "Jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
    "Jodhpur": {"lat": 26.2389, "lon": 73.0243, "state": "Rajasthan"},
    "Kota": {"lat": 25.2138, "lon": 75.8648, "state": "Rajasthan"},
    "Bikaner": {"lat": 28.0229, "lon": 73.3119, "state": "Rajasthan"},
    "Ajmer": {"lat": 26.4499, "lon": 74.6399, "state": "Rajasthan"},
    "Udaipur": {"lat": 24.5854, "lon": 73.7125, "state": "Rajasthan"},
    "Sikar": {"lat": 27.6120, "lon": 75.1397, "state": "Rajasthan"},
    "Bhilwara": {"lat": 25.3407, "lon": 74.6313, "state": "Rajasthan"},
    "Alwar": {"lat": 27.5530, "lon": 76.6346, "state": "Rajasthan"},
    "Sri Ganganagar": {"lat": 29.9094, "lon": 73.8795, "state": "Rajasthan"},
    "Bharatpur": {"lat": 27.2152, "lon": 77.5030, "state": "Rajasthan"},
    "Pali": {"lat": 25.7781, "lon": 73.3311, "state": "Rajasthan"},
    "Barmer": {"lat": 25.7325, "lon": 71.3936, "state": "Rajasthan"},
    
    # Sikkim
    "Gangtok": {"lat": 27.3389, "lon": 88.6065, "state": "Sikkim"},
    "Namchi": {"lat": 27.1670, "lon": 88.3640, "state": "Sikkim"},
    "Gyalshing": {"lat": 27.2833, "lon": 88.2667, "state": "Sikkim"},
    "Mangan": {"lat": 27.5083, "lon": 88.5325, "state": "Sikkim"},
    
    # Tamil Nadu
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
    "Coimbatore": {"lat": 11.0168, "lon": 76.9558, "state": "Tamil Nadu"},
    "Madurai": {"lat": 9.9252, "lon": 78.1198, "state": "Tamil Nadu"},
    "Tiruchirappalli": {"lat": 10.7905, "lon": 78.7047, "state": "Tamil Nadu"},
    "Salem": {"lat": 11.6643, "lon": 78.1460, "state": "Tamil Nadu"},
    "Tirunelveli": {"lat": 8.7139, "lon": 77.7567, "state": "Tamil Nadu"},
    "Tiruppur": {"lat": 11.1085, "lon": 77.3411, "state": "Tamil Nadu"},
    "Erode": {"lat": 11.3410, "lon": 77.7172, "state": "Tamil Nadu"},
    "Vellore": {"lat": 12.9165, "lon": 79.1325, "state": "Tamil Nadu"},
    "Thoothukudi": {"lat": 8.7642, "lon": 78.1348, "state": "Tamil Nadu"},
    "Dindigul": {"lat": 10.3624, "lon": 77.9695, "state": "Tamil Nadu"},
    "Thanjavur": {"lat": 10.7870, "lon": 79.1378, "state": "Tamil Nadu"},
    "Ranipet": {"lat": 12.9487, "lon": 79.3192, "state": "Tamil Nadu"},
    "Sivakasi": {"lat": 9.4533, "lon": 77.8026, "state": "Tamil Nadu"},
    "Karur": {"lat": 10.9601, "lon": 78.0766, "state": "Tamil Nadu"},
    "Udhagamandalam": {"lat": 11.4102, "lon": 76.6950, "state": "Tamil Nadu"},
    
    # Telangana
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
    "Warangal": {"lat": 17.9689, "lon": 79.5941, "state": "Telangana"},
    "Nizamabad": {"lat": 18.6725, "lon": 78.0941, "state": "Telangana"},
    "Karimnagar": {"lat": 18.4386, "lon": 79.1288, "state": "Telangana"},
    "Khammam": {"lat": 17.2473, "lon": 80.1514, "state": "Telangana"},
    "Ramagundam": {"lat": 18.7500, "lon": 79.5000, "state": "Telangana"},
    "Mahbubnagar": {"lat": 16.7375, "lon": 77.9858, "state": "Telangana"},
    "Nalgonda": {"lat": 17.0575, "lon": 79.2675, "state": "Telangana"},
    "Adilabad": {"lat": 19.6640, "lon": 78.5320, "state": "Telangana"},
    "Secunderabad": {"lat": 17.4399, "lon": 78.4983, "state": "Telangana"},
    
    # Tripura
    "Agartala": {"lat": 23.8315, "lon": 91.2868, "state": "Tripura"},
    "Dharmanagar": {"lat": 24.3700, "lon": 92.1700, "state": "Tripura"},
    "Udaipur": {"lat": 23.5333, "lon": 91.4833, "state": "Tripura"},
    "Kailasahar": {"lat": 24.3300, "lon": 92.0000, "state": "Tripura"},
    
    # Uttar Pradesh
    "Lucknow": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh"},
    "Kanpur": {"lat": 26.4499, "lon": 80.3319, "state": "Uttar Pradesh"},
    "Ghaziabad": {"lat": 28.6692, "lon": 77.4538, "state": "Uttar Pradesh"},
    "Varanasi": {"lat": 25.3176, "lon": 82.9739, "state": "Uttar Pradesh"},
    "Meerut": {"lat": 28.9845, "lon": 77.7064, "state": "Uttar Pradesh"},
    "Agra": {"lat": 27.1767, "lon": 78.0081, "state": "Uttar Pradesh"},
    "Prayagraj": {"lat": 25.4358, "lon": 81.8463, "state": "Uttar Pradesh"},
    "Aligarh": {"lat": 27.8974, "lon": 78.0880, "state": "Uttar Pradesh"},
    "Bareilly": {"lat": 28.3670, "lon": 79.4304, "state": "Uttar Pradesh"},
    "Moradabad": {"lat": 28.8386, "lon": 78.7733, "state": "Uttar Pradesh"},
    "Saharanpur": {"lat": 29.9640, "lon": 77.5460, "state": "Uttar Pradesh"},
    "Gorakhpur": {"lat": 26.7605, "lon": 83.3731, "state": "Uttar Pradesh"},
    "Noida": {"lat": 28.5355, "lon": 77.3910, "state": "Uttar Pradesh"},
    "Firozabad": {"lat": 27.1592, "lon": 78.3957, "state": "Uttar Pradesh"},
    "Jhansi": {"lat": 25.4484, "lon": 78.5685, "state": "Uttar Pradesh"},
    "Muzaffarnagar": {"lat": 29.4727, "lon": 77.7085, "state": "Uttar Pradesh"},
    "Mathura": {"lat": 27.4924, "lon": 77.6737, "state": "Uttar Pradesh"},
    "Budaun": {"lat": 28.0376, "lon": 79.1237, "state": "Uttar Pradesh"},
    "Rampur": {"lat": 28.8087, "lon": 79.0259, "state": "Uttar Pradesh"},
    "Shahjahanpur": {"lat": 27.8808, "lon": 79.9128, "state": "Uttar Pradesh"},
    
    # Uttarakhand
    "Dehradun": {"lat": 30.3165, "lon": 78.0322, "state": "Uttarakhand"},
    "Haridwar": {"lat": 29.9457, "lon": 78.1642, "state": "Uttarakhand"},
    "Roorkee": {"lat": 29.8543, "lon": 77.8880, "state": "Uttarakhand"},
    "Haldwani": {"lat": 29.2183, "lon": 79.5130, "state": "Uttarakhand"},
    "Rudrapur": {"lat": 28.9875, "lon": 79.4141, "state": "Uttarakhand"},
    "Kashipur": {"lat": 29.2104, "lon": 78.9620, "state": "Uttarakhand"},
    "Rishikesh": {"lat": 30.0869, "lon": 78.2676, "state": "Uttarakhand"},
    "Nainital": {"lat": 29.3919, "lon": 79.4542, "state": "Uttarakhand"},
    
    # West Bengal
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
    "Siliguri": {"lat": 26.7271, "lon": 88.3953, "state": "West Bengal"},
    "Asansol": {"lat": 23.6889, "lon": 86.9661, "state": "West Bengal"},
    "Durgapur": {"lat": 23.5204, "lon": 87.3119, "state": "West Bengal"},
    "Bardhaman": {"lat": 23.2324, "lon": 87.8614, "state": "West Bengal"},
    "Malda": {"lat": 25.0220, "lon": 88.1420, "state": "West Bengal"},
    "Baharampur": {"lat": 24.1009, "lon": 88.2508, "state": "West Bengal"},
    "Habra": {"lat": 22.8300, "lon": 88.6500, "state": "West Bengal"},
    "Jalpaiguri": {"lat": 26.5200, "lon": 88.7200, "state": "West Bengal"},
    "Kharagpur": {"lat": 22.3300, "lon": 87.3200, "state": "West Bengal"},
    "Darjeeling": {"lat": 27.0380, "lon": 88.2627, "state": "West Bengal"},
    "Howrah": {"lat": 22.5958, "lon": 88.2636, "state": "West Bengal"},
    "Cooch Behar": {"lat": 26.3200, "lon": 89.4200, "state": "West Bengal"},
    
    # Union Territories
    "Port Blair": {"lat": 11.6234, "lon": 92.7265, "state": "Andaman and Nicobar Islands"},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794, "state": "Chandigarh"},
    "Silvassa": {"lat": 20.2667, "lon": 73.0167, "state": "Dadra and Nagar Haveli"},
    "Daman": {"lat": 20.4283, "lon": 72.8397, "state": "Daman and Diu"},
    "Diu": {"lat": 20.7144, "lon": 70.9874, "state": "Daman and Diu"},
    "Kavaratti": {"lat": 10.5593, "lon": 72.6358, "state": "Lakshadweep"},
    "Pondicherry": {"lat": 11.9416, "lon": 79.8083, "state": "Puducherry"},
    "Karaikal": {"lat": 10.9254, "lon": 79.8380, "state": "Puducherry"},
    "Yanam": {"lat": 16.7271, "lon": 82.2176, "state": "Puducherry"},
    "Mahe": {"lat": 11.7018, "lon": 75.5356, "state": "Puducherry"},
    "Leh": {"lat": 34.1526, "lon": 77.5771, "state": "Ladakh"},
    "Kargil": {"lat": 34.5539, "lon": 76.1349, "state": "Ladakh"}
}

# Pollutants and their descriptions
POLLUTANTS = {
    "PM2.5": "Fine particulate matter with diameter less than 2.5 micrometers. Can penetrate deep into the lungs and even enter the bloodstream.",
    "PM10": "Particulate matter with diameter less than 10 micrometers. Can penetrate into the lungs and cause respiratory issues.",
    "NO2": "Nitrogen Dioxide, a gas produced by burning fuel. Can cause inflammation of the airways and respiratory problems.",
    "SO2": "Sulfur Dioxide, produced from burning fossil fuels. Can cause respiratory issues and contribute to acid rain.",
    "CO": "Carbon Monoxide, a poisonous gas produced by incomplete combustion. Reduces oxygen delivery to the body's organs.",
    "O3": "Ozone, a harmful air pollutant at ground level. Can trigger health problems like chest pain, coughing, and throat irritation."
}

# CPCB API Endpoint
CPCB_API_ENDPOINT = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
CPCB_HISTORICAL_ENDPOINT = "https://api.data.gov.in/resource/fd384a41-d450-4a26-8a5d-b8a7d218fb97"

# Health recommendations based on AQI levels
HEALTH_RECOMMENDATIONS = {
    "Good": "Air quality is good. Perfect for outdoor activities.",
    "Satisfactory": "Air quality is acceptable. Enjoy outdoor activities.",
    "Moderate": "Consider reducing prolonged outdoor exertion if you're sensitive to air pollution.",
    "Poor": "People with heart or lung disease, older adults, and children should reduce prolonged exertion.",
    "Very Poor": "Everyone should reduce prolonged exertion. People with heart or lung disease, older adults, and children should avoid all outdoor activities.",
    "Severe": "Everyone should avoid all outdoor activities. People with heart or lung disease, older adults, and children should remain indoors and keep activity levels low."
}

# Default prediction days
DEFAULT_PREDICTION_DAYS = 7

# Define map zoom level
DEFAULT_MAP_ZOOM = 5

# List of all Indian states and union territories
INDIAN_STATES = [
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Delhi",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jammu & Kashmir",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
    "Andaman and Nicobar Islands",
    "Chandigarh",
    "Dadra and Nagar Haveli",
    "Daman and Diu",
    "Lakshadweep",
    "Puducherry",
    "Ladakh"
]
