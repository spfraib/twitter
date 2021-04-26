import numpy as np
from pyspark.sql.functions import lit, col, udf


params_dict = {'is_hired_1mo': {'jan5_iter0': {'params': [[723.2620805933465,
     -702.2451911199508],
    [613.4486972048122, -595.1196307960936],
    [733.258141055328, -712.2006428170077],
    [661.3329464067762, -641.7128024023933],
    [1044.2132361447425, -1014.3412335461968],
    [760.7305094925329, -738.2674301229783],
    [1223.2839270655493, -1187.8885718938882],
    [1411.3692429265961, -1369.2535239001438],
    [915.7284300356555, -889.5250879004069],
    [604.535142042308, -586.3917627260167]]},
  'feb22_iter1': {'params': [[16.40686649614705, -14.677977076857104],
    [591.5076467860082, -582.7429231901958],
    [20.212542711637102, -18.893473286371744],
    [30.972651453352345, -28.546026279605293],
    [99.11333945601051, -96.23451969008126],
    [3264.899764501499, -3215.582019864592],
    [20.67887340879628, -19.29108690567077],
    [25.767649620173266, -23.609515967163713],
    [33.31041294622334, -31.44396095058162],
    [12.788363642411957, -11.760780739174509]]},
  'feb23_iter2': {'params': [[24.69125802040602, -21.191885989723193],
    [40.171160839429334, -35.34816816247577],
    [15.999599518702734, -12.996886890618203],
    [39.25824690139436, -34.610638109365496],
    [18.306522512277414, -15.156048342856339],
    [19.823568156979423, -16.527447439301895],
    [23.635440012563908, -19.581999983838177],
    [14.207256720296149, -10.68830884701004],
    [35.79683517217701, -31.35723304241782],
    [16.114940738663925, -13.027857588627795]]},
  'feb25_iter3': {'params': [[12.24270982597547, -10.587795051063157],
    [29.659161688830405, -27.20241412920571],
    [15.008223485186178, -13.018016825542716],
    [24.144179698072595, -22.576109395500506],
    [18.58966830580949, -16.591890612206434],
    [31.20522971833408, -29.041487847487844],
    [11.138220770281155, -10.069704627583201],
    [15.361969835360096, -13.695924563535511],
    [135.06105682914796, -131.49693195650477],
    [17.20557845217599, -15.82679498311033]]},
  'mar1_iter4': {'params': [[6.612720846705297, -2.961006635765658],
    [5.756149012621954, -3.2152941175991216],
    [7.731789906510983, -4.14675035585539],
    [7.158252135220039, -3.025878328627267],
    [8.059857643268137, -3.135723102725056],
    [5.949845934507181, -3.169673708309554],
    [7.622412788635907, -2.2006118146121185],
    [6.703667083561221, -2.977283559544276],
    [5.170304052390332, -2.430051774430339],
    [5.678260753371807, -3.0672190478988095]]}},
 'is_unemployed': {'jan5_iter0': {'params': [[954.3591890595086,
     -941.4691482652067],
    [1313.385337788351, -1297.6953586205932],
    [31747.40740927471, -31362.817139946234],
    [1197.896490110487, -1182.7880493392825],
    [1585.769686428507, -1565.8649943054509],
    [767.3604953620211, -756.9785117006832],
    [968.8623129044793, -957.8319364747919],
    [848.5806374750873, -838.3401869582777],
    [653.5562577919832, -645.6433348670768],
    [1758.7551682607555, -1736.2849273898805]]},
  'feb22_iter1': {'params': [[359.5191905400591, -341.6155676750235],
    [155.0529082876009, -146.82392869337832],
    [290.8835138664372, -276.4119682413097],
    [341.9126376072391, -324.91886257232227],
    [267.8949941237716, -254.68364871467602],
    [297.8733293161619, -283.414063960708],
    [441.7168969506824, -420.636643759353],
    [332.16165020585527, -316.06435066136703],
    [175.35747193832307, -165.81511998576624],
    [242.69644405522774, -231.04763203182196]]},
  'feb23_iter2': {'params': [[1309743.3863048458, -1250740.212005921],
    [1206677.4932503635, -1152317.5160356406],
    [466.25537381245493, -434.39175641246766],
    [1015783.7779773731, -970021.3898043218],
    [326390.0105933796, -311682.0623584801],
    [2314153.4286632384, -2209911.612559896],
    [1108584.8044864126, -1058643.2906551438],
    [2503925.8038917757, -2391134.8731667837],
    [556307.5923878539, -531244.0629980547],
    [1934248.2700114322, -1847118.9713616052]]},
  'feb25_iter3': {'params': [[233.61788927872126, -231.708898963589],
    [93.44115568446003, -92.07487042444568],
    [158.01078949202116, -156.20246964279892],
    [156.01834625860863, -154.1445229901904],
    [126.65648053724071, -125.84257546570582],
    [123.78763234133746, -122.64637389851879],
    [70.71491953806678, -69.69041185525813],
    [124.94076166137827, -123.34908752154368],
    [148.61489250153474, -147.51854044141078],
    [122.68076689793658, -121.45584476326239]]},
  'mar1_iter4': {'params': [[146.10025379080042, -142.92287203128456],
    [7.775408397062544, -6.578540598829755],
    [161.77388648130758, -157.86468721811715],
    [95.0033752320291, -91.89914097880549],
    [9.1168480176437, -7.2388928450741865],
    [16.307818096927832, -14.602610183128762],
    [15.60018272190997, -14.133370525315431],
    [9.293804065749303, -7.936193991211735],
    [10.462741907360936, -8.804571429577216],
    [147.2779406277176, -143.83692817496564]]}},
 'lost_job_1mo': {'jan5_iter0': {'params': [[1806.136358238624,
     -1798.406562163408],
    [12852.72183301292, -12784.742048728636],
    [14009.574367044097, -13935.423899265916],
    [6875.083556925325, -6837.19793646928],
    [6542.734224205343, -6507.0004892149],
    [7961.511395542616, -7927.497546542947],
    [7294.718558093769, -7263.552356022973],
    [915146.5053955431, -911424.7277082093],
    [394.3243929118417, -390.58761402803606],
    [713.3629790968135, -711.0685655239383]]},
  'feb22_iter1': {'params': [[5.796512612539675, -2.5949723332546792],
    [13.900674817839919, -4.296185546357563],
    [12.407314549300128, -4.840326685472993],
    [12.739115185575132, -4.817240862843977],
    [160.59475491510426, -62.73431488848247],
    [16.735636811499393, -5.6527348661494115],
    [9.408385145266287, -3.4226197734407036],
    [21.960347117347286, -7.177610972434149],
    [11.550990227103549, -4.265032194987646],
    [12.091648004619199, -4.410587958124836]]},
  'feb23_iter2': {'params': [[2149.494175769004, -2140.9851408882078],
    [2001.771939876708, -1994.9729775891824],
    [2615.535584655213, -2604.9944309871394],
    [55520.35864671264, -55353.462329776754],
    [2471.4141551480907, -2463.022835225955],
    [42223.0182139652, -42088.425817642885],
    [20002.491779184566, -19939.068545290174],
    [2934.7061665942747, -2925.0991689381995],
    [27292.08622123958, -27196.742525765654],
    [6750.605255457174, -6729.318142264537]]},
  'feb25_iter3': {'params': [[38.87985659632536, -37.41612344401107],
    [54.54073834775329, -51.905187641460685],
    [80.65291126202314, -78.79177395440455],
    [64.51073023993503, -63.95790827088694],
    [5.641008509073449, -4.891150571402365],
    [7.161542080593472, -6.239454704976834],
    [157.80232434237172, -156.26076868438793],
    [47.62324176996915, -46.37581147941766],
    [43.99436909813125, -41.18253996478061],
    [6.524005090979128, -5.89340109224047]]},
  'mar1_iter4': {'params': [[20.117283375481346, -17.21204954330244],
    [35.64198122077081, -30.93553000088976],
    [9.557553035363581, -7.332878786393763],
    [10.667377068590811, -8.16526226859789],
    [9.90275887768058, -6.793926683075684],
    [23.024195199508192, -19.377988006022004],
    [79.70569228194496, -73.33329363965657],
    [9.088344427861458, -6.9778650561535835],
    [5.9292514249170685, -4.31746543465311],
    [10.944875145949378, -8.78797613927884]]}},
 'job_search': {'jan5_iter0': {'params': [[4927.794640401552,
     -4915.221993483281],
    [3054.7186530281715, -3046.6570424689316],
    [3280.1122939581523, -3270.883362383758],
    [8012.224781639475, -7992.360949467039],
    [7455.606076652913, -7436.068863486747],
    [4970.997683136551, -4958.472946493052],
    [4233.122948061402, -4221.864409532163],
    [2484.3122423128325, -2477.329994306285],
    [2401.9381227594126, -2394.891631676984],
    [4040.1619201641506, -4029.8726673861947]]},
  'feb22_iter1': {'params': [[218.6338104295803, -215.85576762283537],
    [503.3956433248866, -498.2835539178315],
    [380.2950504351187, -376.32183692403396],
    [295.3905804208253, -292.06556058210316],
    [267.30965844863215, -264.129505354175],
    [505.7909907193474, -501.01937141453993],
    [464.78383565662784, -460.10357384050604],
    [592.8660684531328, -587.7470487664253],
    [186.07262632257041, -183.75320190226748],
    [307.115524356046, -303.4079450664146]]},
  'feb23_iter2': {'params': [[215.2805649196513, -212.96703587513],
    [215.7135558803261, -213.8697935274163],
    [202.83243880221377, -201.1773837068967],
    [223.91716608871127, -221.90601869425274],
    [207.6900593701977, -205.57464336517157],
    [261.2807385796726, -259.263108592236],
    [236.24398342688207, -233.93967382525912],
    [231.64196837500637, -229.65054110490385],
    [142.00023465025592, -140.4470569949443],
    [254.4201933629614, -252.29565325200193]]},
  'feb25_iter3': {'params': [[19.102405760351992, -15.517324570379136],
    [19.681239620545096, -14.843818372330478],
    [9.474633552980576, -6.6179488790952945],
    [134.25804282084664, -128.76926198390046],
    [7.628146163159228, -4.914834400057528],
    [14.366584656146347, -10.636108424740396],
    [22.268587930347795, -17.935456118044044],
    [9.33831041220166, -6.2653641076761435],
    [19.793605139455337, -16.496889293353163],
    [13.52634962421381, -10.45870440163934]]},
  'mar1_iter4': {'params': [[495.64163581956484, -491.1427836402967],
    [959.1281867017484, -952.263051533974],
    [1511.5267584896383, -1502.399277676573],
    [1021.7111457204306, -1015.2373174714252],
    [1323.03263694925, -1314.3203093388381],
    [1783.2601053544618, -1772.346108882602],
    [899.7912338630185, -893.1471101427734],
    [777.0662701435057, -771.0868737567023],
    [574.5070685488189, -569.5417858301114],
    [1540.270419076618, -1531.0563087055034]]}},
 'job_offer': {'jan5_iter0': {'params': [[15553.612194837868,
     -15491.25160900457],
    [34522110.9335532, -34379917.450467974],
    [18888.662629399747, -18812.470363896355],
    [99601281.4511164, -99191031.39795059],
    [23769034.182490308, -23671132.154396635],
    [38489587.82601724, -38331052.84069898],
    [27431368.616737686, -27318382.64846576],
    [17123.389161704337, -17053.552146090504],
    [62908548.31695484, -62649433.92645448],
    [69406897.74668305, -69121016.33202283]]},
  'feb22_iter1': {'params': [[112367.31283020851, -112232.78619102515],
    [75408.98615703719, -75319.17352011033],
    [87414.36124132547, -87310.36290354002],
    [73843.96830197144, -73756.84492346883],
    [79321.17477422723, -79226.82086949408],
    [99165.47473532146, -99047.14956326492],
    [68493.20353861144, -68411.33453246819],
    [62285.858172235, -62213.180247955235],
    [73969.59410407257, -73882.31994823448],
    [59919.914059414215, -59848.18049043094]]},
  'feb23_iter2': {'params': [[1596.7048644011682, -1572.7155463960357],
    [3448.644716809183, -3407.970294414705],
    [1470.1295729653873, -1447.3487540892106],
    [1744.5745264498637, -1719.1742288219464],
    [1588.7627090275014, -1565.046606204143],
    [1480.1492882599348, -1457.3513473856788],
    [1577.4930129035952, -1552.5391386125996],
    [1418.3519662039205, -1396.435906229709],
    [1591.533668702584, -1568.0923842516108],
    [1484.1140891228556, -1461.5929830517696]]},
  'feb25_iter3': {'params': [[37.06028124156873, -32.65335640475719],
    [43.558315782286655, -39.375538400329454],
    [855.6431867776665, -832.1509373530188],
    [43.94485075133368, -38.36986603396645],
    [39.899879930419786, -36.320491838120354],
    [864.675197550401, -841.1677091915006],
    [1967493.1241011266, -1915581.763332232],
    [32.833198783777185, -28.820091690206503],
    [29.479368963375872, -25.43874344616801],
    [32.93171769352762, -28.61255560067027]]},
  'mar1_iter4': {'params': [[7532.910989154963, -7502.18358813135],
    [8724.454080211299, -8689.471951325839],
    [47483618.975132965, -47354922.08848605],
    [110293544.93489356, -109994612.76142542],
    [6821.050525893615, -6793.167018046988],
    [74415605.0083998, -74213914.3343166],
    [153646307.7679448, -153229882.61742076],
    [8921.815877330264, -8885.87791171161],
    [52871531.38862602, -52728231.71559521],
    [192571924.95365256, -192050001.09603298]]}}}


folder_dict = {0: ['iter_0-convbert-969622-evaluation', 'jan5_iter0'], 1: ['iter_1-convbert-3050798-evaluation', 'feb22_iter1'], 2:['iter_2-convbert-3134867-evaluation', 'feb23_iter2'], 3:['iter_3-convbert-3174249-evaluation', 'feb25_iter3'], 4:['iter_4-convbert-3297962-evaluation', 'mar1_iter4']}

# def calibrate(score, params):
#     return np.mean([1 / (1 + np.exp(-(param[0] * score + param[1]))) for param in params], axis=0)


path_to_tweets = os.path.join('/user/mt4493/twitter/twitter-labor-data/random_samples/random_samples_splitted', country_code, 'evaluation')
random_df = spark.read.parquet(path_to_tweets)

for label in ['is_hired_1mo', 'lost_job_1mo', 'job_search', 'job_offer', 'is_unemployed']:
    for iter in range(5):
        inference_folder = folder_dict[0][0]
        data_folder = folder_dict[0][1]
        params = params_dict[label][data_folder]['params']
        calibrateUDF = F.udf(lambda x: F.mean([1 / (1 + F.exp(-(param[0] * x + param[1]))) for param in params]), FloatType())
        path_to_scores = os.path.join('/user/mt4493/twitter/twitter-labor-data/inference', country_code, inference_folder, 'output', label)  # Prediction scores from classification
        scores_df = spark.read.parquet(path_to_scores)
        df = random_df.join(scores_df, on='tweet_id', how='inner')
        df = df.withColumn("calibrated_score", calibrateUDF(col("score")))
