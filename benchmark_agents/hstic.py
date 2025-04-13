import random

# Глобальные переменные и инициализация
IsOver = True
IsStart0 = None
Start = None
Start0 = None
MaxSize = 13
Size = 11
IsRunning = False
IsBlinking = False
LastEvent = ""
MoveCount = 0
MaxMoveCount = 0
MaxFld = Size * Size
IsSwap = None
ActiveColor = 0
IsPlayer = [None, None]
Level = [None, None]

# Создаём двумеричные массивы фиксированного максимального размера
ImgNum = [[None for _ in range(MaxSize)] for _ in range(MaxSize)]
Fld = [[0 for _ in range(MaxSize)] for _ in range(MaxSize)]
Pot = [[None for _ in range(MaxSize)] for _ in range(MaxSize)]
for i in range(MaxSize):
    for j in range(MaxSize):
        Pot[i][j] = [None, None, None, None]
Bridge = [[None for _ in range(MaxSize)] for _ in range(MaxSize)]
for i in range(MaxSize):
    for j in range(MaxSize):
        Bridge[i][j] = [None, None, None, None]
Upd = [[None for _ in range(MaxSize)] for _ in range(MaxSize)]
History = [[None, None] for _ in range(MaxFld + 1)]

# Начальные настройки
IsStart0 = True
IsPlayer[0] = True    # Красный: игрок
IsPlayer[1] = False   # Синий: компьютер
Level[0] = 3
Level[1] = 3

# ------------------------------
# Функции, не связанные с UI
# ------------------------------

def SetSize(nn):
    global Size, MaxFld, History, IsRunning, IsBlinking, LastEvent
    if IsRunning or IsBlinking:
        LastEvent += "SetSize(" + str(nn) + ");"
        return
    Size = nn
    MaxFld = Size * Size
    History = [[None, None] for _ in range(MaxFld + 1)]
    Init()

def Init():
    global Fld, MoveCount, MaxMoveCount, IsOver, Start0, IsStart0, Size
    if IsRunning or IsBlinking:
        global LastEvent
        LastEvent += "Init();"
        return
    for ii in range(Size):
        for jj in range(Size):
            Fld[ii][jj] = 0
    Start0 = True if IsStart0 else False
    MoveCount = 0
    MaxMoveCount = 0
    IsOver = False

def SetOption(nn, mm):
    global IsPlayer, IsStart0, IsRunning, IsBlinking, LastEvent
    if IsRunning or IsBlinking:
        LastEvent += "SetOption(" + str(nn) + "," + str(mm) + ");"
        return
    if nn < 2:
        if mm == 0:
            IsPlayer[nn] = True
        else:
            IsPlayer[nn] = False
    else:
        IsStart0 = mm

def SetLevel(nn, mm):
    global Level, IsRunning, IsBlinking, LastEvent
    if IsRunning or IsBlinking:
        LastEvent += "SetLevel(" + str(nn) + "," + str(mm) + ");"
        return
    Level[nn] = mm

def random_js(nn):
    return int(random.random() * 1000) % nn

def SwapTest():
    # Предполагаем, что правило swap всегда активно.
    global MoveCount, Size
    if MoveCount == 0:
        ii = random_js(4)
        jj = random_js(4 - ii)
        if random_js(2) < 1:
            ii = Size - 1 - ii
            jj = Size - 1 - jj
        MakeMove(ii, jj, False)
        return True
    if MoveCount == 1:
        for ii in range(Size):
            for jj in range(Size):
                if Fld[ii][jj] != 0:
                    if (ii + jj < 2) or (ii + jj > 2 * Size - 4):
                        return False
                    if (ii + jj == 2) or (ii + jj == 2 * Size - 4):
                        if random_js(2) < 1:
                            return False
                    MakeMove(ii, jj, False)
                    return True
    return False

def MakeMove(ii, jj, oo):
    global MoveCount, Start0, Fld, History, MaxMoveCount, IsSwap, IsOver
    iis = ii
    jjs = jj
    if MoveCount == 1:
        if Fld[ii][jj] != 0:
            Fld[ii][jj] = 0
            # В оригинале обновление графики удалено
            iis = jj
            jjs = ii
            IsSwap = 1
        else:
            IsSwap = 0
    ccol = ((MoveCount + 1 + (1 if Start0 else 0)) % 2) * 2 - 1
    Fld[iis][jjs] = ccol
    if History[MoveCount][0] != ii:
        History[MoveCount][0] = ii
        MaxMoveCount = MoveCount + 1
    if History[MoveCount][1] != jj:
        History[MoveCount][1] = jj
        MaxMoveCount = MoveCount + 1
    MoveCount += 1
    if MaxMoveCount < MoveCount:
        MaxMoveCount = MoveCount
    if not oo:
        return
    GetPot(0)
    WritePot(True)
    if ccol < 0:
        if (Pot[ii][jj][2] > 0) or (Pot[ii][jj][3] > 0):
            return
        Blink(0)
    else:
        if (Pot[ii][jj][0] > 0) or (Pot[ii][jj][1] > 0):
            return
        Blink(0)
    IsOver = True

def ShowPot():
    pass

def RedPotCol(vv):
    return "#ff0000"

def BluePotCol(vv):
    return "#0000ff"

def WritePot(bb):
    pass

def sign(xx):
    if xx < 0:
        return -1
    if xx > 0:
        return 1
    return 0

LastAIMove = None

def GetBestMove(theCol, theLevel):
    global MoveCount, Size, Fld, Pot, Bridge, LastAIMove, IsRunning, IsOver
    ii_b = None
    jj_b = None
    ff = 0
    ii_q = 0
    jj_q = 0
    vv = {}
    if MoveCount > 0:
        ff = 190 / (MoveCount * MoveCount)
    mm = 20000
    for ii in range(Size):
        for jj in range(Size):
            if Fld[ii][jj] != 0:
                ii_q += 2 * ii + 1 - Size
                jj_q += 2 * jj + 1 - Size
    ii_q = sign(ii_q)
    jj_q = sign(jj_q)
    for ii in range(Size):
        for jj in range(Size):
            if Fld[ii][jj] == 0:
                mmp = random.random() * (49 - theLevel * 16)
                mmp += (abs(ii - 5) + abs(jj - 5)) * ff
                mmp += 8 * (ii_q * (ii - 5) + jj_q * (jj - 5)) / (MoveCount + 1)
                if theLevel > 2:
                    for kk in range(4):
                        mmp -= Bridge[ii][jj][kk]
                pp0 = Pot[ii][jj][0] + Pot[ii][jj][1]
                pp1 = Pot[ii][jj][2] + Pot[ii][jj][3]
                mmp += pp0 + pp1
                if (pp0 <= 268) or (pp1 <= 268):
                    mmp -= 400
                vv[ii * Size + jj] = mmp
                if mmp < mm:
                    mm = mmp
                    ii_b = ii
                    jj_b = jj
    if theLevel > 2:
        mm += 108
        for ii in range(Size):
            for jj in range(Size):
                key = ii * Size + jj
                if key in vv and vv[key] < mm:
                    if theCol < 0:  # красный
                        if (ii > 3) and (ii < Size - 1) and (jj > 0) and (jj < 3):
                            if Fld[ii - 1][jj + 2] == -theCol:
                                cc = CanConnectFarBorder(ii - 1, jj + 2, -theCol)
                                if cc < 2:
                                    ii_b = ii
                                    if cc < -1:
                                        ii_b -= 1
                                        cc += 1
                                    jj_b = jj - cc
                                    mm = vv[key]
                        if (ii > 0) and (ii < Size - 1) and (jj == 0):
                            if (Fld[ii - 1][jj + 2] == -theCol and
                                Fld[ii - 1][jj] == 0 and Fld[ii - 1][jj + 1] == 0 and
                                Fld[ii][jj + 1] == 0 and Fld[ii + 1][jj] == 0):
                                ii_b = ii; jj_b = jj; mm = vv[key]
                        if (ii > 0) and (ii < Size - 4) and (jj < Size - 1) and (jj > Size - 4):
                            if Fld[ii + 1][jj - 2] == -theCol:
                                cc = CanConnectFarBorder(ii + 1, jj - 2, -theCol)
                                if cc < 2:
                                    ii_b = ii
                                    if cc < -1:
                                        ii_b += 1
                                        cc += 1
                                    jj_b = jj + cc
                                    mm = vv[key]
                        if (ii > 0) and (ii < Size - 1) and (jj == Size - 1):
                            if (Fld[ii + 1][jj - 2] == -theCol and
                                Fld[ii + 1][jj] == 0 and Fld[ii + 1][jj - 1] == 0 and
                                Fld[ii][jj - 1] == 0 and Fld[ii - 1][jj] == 0):
                                ii_b = ii; jj_b = jj; mm = vv[key]
                    else:
                        if (jj > 3) and (jj < Size - 1) and (ii > 0) and (ii < 3):
                            if Fld[ii + 2][jj - 1] == -theCol:
                                cc = CanConnectFarBorder(ii + 2, jj - 1, -theCol)
                                if cc < 2:
                                    jj_b = jj
                                    if cc < -1:
                                        jj_b -= 1
                                        cc += 1
                                    ii_b = ii - cc
                                    mm = vv[key]
                        if (jj > 0) and (jj < Size - 1) and (ii == 0):
                            if (Fld[ii + 2][jj - 1] == -theCol and
                                Fld[ii][jj - 1] == 0 and Fld[ii + 1][jj - 1] == 0 and
                                Fld[ii + 1][jj] == 0 and Fld[ii][jj + 1] == 0):
                                ii_b = ii; jj_b = jj; mm = vv[key]
                        if (jj > 0) and (jj < Size - 4) and (ii < Size - 1) and (ii > Size - 4):
                            if Fld[ii - 2][jj + 1] == -theCol:
                                cc = CanConnectFarBorder(ii - 2, jj + 1, -theCol)
                                if cc < 2:
                                    jj_b = jj
                                    if cc < -1:
                                        jj_b += 1
                                        cc += 1
                                    ii_b = ii + cc
                                    mm = vv[key]
                        if (jj > 0) and (jj < Size - 1) and (ii == Size - 1):
                            if (Fld[ii - 2][jj + 1] == -theCol and
                                Fld[ii][jj + 1] == 0 and Fld[ii - 1][jj + 1] == 0 and
                                Fld[ii - 1][jj] == 0 and Fld[ii][jj - 1] == 0):
                                ii_b = ii; jj_b = jj; mm = vv[key]
    MakeMove(ii_b, jj_b, False)
    global LastAIMove
    LastAIMove = {'i': ii_b, 'j': jj_b}
    IsRunning = False
    if theCol < 0:
        if (Pot[ii_b][jj_b][2] > 140) or (Pot[ii_b][jj_b][3] > 140):
            WritePot(False)
            return
        Blink(-2)
    else:
        if (Pot[ii_b][jj_b][0] > 140) or (Pot[ii_b][jj_b][1] > 140):
            WritePot(False)
            return
        Blink(-2)
    IsOver = True

def CanConnectFarBorder(nn, mm, cc):
    global Size, Fld
    if cc > 0:  # blue
        if 2 * mm < Size - 1:
            for ii in range(Size):
                for jj in range(mm):
                    if (jj - ii < mm - nn) and (ii + jj <= nn + mm) and (Fld[ii][jj] != 0):
                        return 2
            if Fld[nn - 1][mm] == -cc:
                return 0
            if Fld[nn - 1][mm - 1] == -cc:
                if GetFld(nn + 2, mm - 1) == -cc:
                    return 0
                return -1
            if GetFld(nn + 2, mm - 1) == -cc:
                return -2
        else:
            for ii in range(Size):
                for jj in range(Size - 1, mm, -1):
                    if (jj - ii > mm - nn) and (ii + jj >= nn + mm) and (Fld[ii][jj] != 0):
                        return 2
            if Fld[nn + 1][mm] == -cc:
                return 0
            if Fld[nn + 1][mm + 1] == -cc:
                if GetFld(nn - 2, mm + 1) == -cc:
                    return 0
                return -1
            if GetFld(nn - 2, mm + 1) == -cc:
                return -2
    else:
        if 2 * nn < Size - 1:
            for jj in range(Size):
                for ii in range(nn):
                    if (ii - jj < nn - mm) and (ii + jj <= nn + mm) and (Fld[ii][jj] != 0):
                        return 2
            if Fld[nn][mm - 1] == -cc:
                return 0
            if Fld[nn - 1][mm - 1] == -cc:
                if GetFld(nn - 1, mm + 2) == -cc:
                    return 0
                return -1
            if GetFld(nn - 1, mm + 2) == -cc:
                return -2
        else:
            for jj in range(Size):
                for ii in range(Size - 1, nn, -1):
                    if (ii - jj > nn - mm) and (ii + jj >= nn + mm) and (Fld[ii][jj] != 0):
                        return 2
            if Fld[nn][mm + 1] == -cc:
                return 0
            if Fld[nn + 1][mm + 1] == -cc:
                if GetFld(nn + 1, mm - 2) == -cc:
                    return 0
                return -1
            if GetFld(nn + 1, mm - 2) == -cc:
                return -2
    return 1

def GetFld(ii, jj):
    global Fld, Size
    if ii < 0:
        return -1
    if jj < 0:
        return 1
    if ii >= Size:
        return -1
    if jj >= Size:
        return 1
    return Fld[ii][jj]

def Blink(nn):
    # Заглушка – никаких анимаций
    pass

def GetPot(llevel):
    global ActiveColor, MoveCount, Start0, Size, Fld, Pot, Bridge, Upd
    dd = 128
    ActiveColor = ((MoveCount + 1 + (1 if Start0 else 0)) % 2) * 2 - 1
    for ii in range(Size):
        for jj in range(Size):
            for kk in range(4):
                Pot[ii][jj][kk] = 20000
                Bridge[ii][jj][kk] = 0
    for ii in range(Size):
        if Fld[ii][0] == 0:
            Pot[ii][0][0] = dd
        else:
            if Fld[ii][0] > 0:
                Pot[ii][0][0] = 0
        if Fld[ii][Size - 1] == 0:
            Pot[ii][Size - 1][1] = dd
        else:
            if Fld[ii][Size - 1] > 0:
                Pot[ii][Size - 1][1] = 0
    for jj in range(Size):
        if Fld[0][jj] == 0:
            Pot[0][jj][2] = dd
        else:
            if Fld[0][jj] < 0:
                Pot[0][jj][2] = 0
        if Fld[Size - 1][jj] == 0:
            Pot[Size - 1][jj][3] = dd
        else:
            if Fld[Size - 1][jj] < 0:
                Pot[Size - 1][jj][3] = 0
    for kk in range(2):
        for ii in range(Size):
            for jj in range(Size):
                Upd[ii][jj] = True
        nn_iter = 0
        while True:
            nn_iter += 1
            bb = 0
            for ii in range(Size):
                for jj in range(Size):
                    if Upd[ii][jj]:
                        bb += SetPot(ii, jj, kk, 1, llevel)
            for ii in range(Size - 1, -1, -1):
                for jj in range(Size - 1, -1, -1):
                    if Upd[ii][jj]:
                        bb += SetPot(ii, jj, kk, 1, llevel)
            if not (bb > 0 and nn_iter < 12):
                break
    for kk in range(2, 4):
        for ii in range(Size):
            for jj in range(Size):
                Upd[ii][jj] = True
        nn_iter = 0
        while True:
            nn_iter += 1
            bb = 0
            for ii in range(Size):
                for jj in range(Size):
                    if Upd[ii][jj]:
                        bb += SetPot(ii, jj, kk, -1, llevel)
            for ii in range(Size - 1, -1, -1):
                for jj in range(Size - 1, -1, -1):
                    if Upd[ii][jj]:
                        bb += SetPot(ii, jj, kk, -1, llevel)
            if not (bb > 0 and nn_iter < 12):
                break

def SetPot(ii, jj, kk, cc, llevel):
    global Upd, Bridge, Fld, Pot, Size, ActiveColor
    Upd[ii][jj] = False
    Bridge[ii][jj][kk] = 0
    if Fld[ii][jj] == -cc:
        return 0
    ddb = 0
    oo = 0
    dd = 140
    bb_val = 66
    if cc != ActiveColor:
        bb_val = 52
    vv = [None] * 6
    tt = [None] * 6
    vv[0] = PotVal(ii + 1, jj, kk, cc)
    vv[1] = PotVal(ii, jj + 1, kk, cc)
    vv[2] = PotVal(ii - 1, jj + 1, kk, cc)
    vv[3] = PotVal(ii - 1, jj, kk, cc)
    vv[4] = PotVal(ii, jj - 1, kk, cc)
    vv[5] = PotVal(ii + 1, jj - 1, kk, cc)
    for ll in range(6):
        if (vv[ll] >= 30000) and (vv[(ll + 2) % 6] >= 30000):
            if vv[(ll + 1) % 6] < 0:
                ddb = +32
            else:
                vv[(ll + 1) % 6] += 128
    for ll in range(6):
        if (vv[ll] >= 30000) and (vv[(ll + 3) % 6] >= 30000):
            ddb += 30
    mm = 30000
    for ll in range(6):
        if vv[ll] < 0:
            vv[ll] += 30000
            tt[ll] = 10
        else:
            tt[ll] = 1
        if mm > vv[ll]:
            mm = vv[ll]
    nn_val = 0
    for ll in range(6):
        if vv[ll] == mm:
            nn_val += tt[ll]
    if llevel > 1:
        Bridge[ii][jj][kk] = nn_val / 5
        if (nn_val >= 2) and (nn_val < 10):
            Bridge[ii][jj][kk] = bb_val + nn_val - 2
            mm -= 32
        if nn_val < 2:
            oo = 30000
            for ll in range(6):
                if (vv[ll] > mm) and (oo > vv[ll]):
                    oo = vv[ll]
            if oo <= mm + 104:
                Bridge[ii][jj][kk] = bb_val - (oo - mm) / 4
                mm -= 64
            mm += oo
            mm = mm / 2
    if (ii > 0) and (ii < Size - 1) and (jj > 0) and (jj < Size - 1):
        Bridge[ii][jj][kk] += ddb
    else:
        Bridge[ii][jj][kk] -= 2
    if ((ii == 0) or (ii == Size - 1)) and ((jj == 0) or (jj == Size - 1)):
        Bridge[ii][jj][kk] /= 2
    if Bridge[ii][jj][kk] > 68:
        Bridge[ii][jj][kk] = 68
    if Fld[ii][jj] == cc:
        if mm < Pot[ii][jj][kk]:
            Pot[ii][jj][kk] = mm
            SetUpd(ii + 1, jj, cc)
            SetUpd(ii, jj + 1, cc)
            SetUpd(ii - 1, jj + 1, cc)
            SetUpd(ii - 1, jj, cc)
            SetUpd(ii, jj - 1, cc)
            SetUpd(ii + 1, jj - 1, cc)
            return 1
        return 0
    if mm + dd < Pot[ii][jj][kk]:
        Pot[ii][jj][kk] = mm + dd
        SetUpd(ii + 1, jj, cc)
        SetUpd(ii, jj + 1, cc)
        SetUpd(ii - 1, jj + 1, cc)
        SetUpd(ii - 1, jj, cc)
        SetUpd(ii, jj - 1, cc)
        SetUpd(ii + 1, jj - 1, cc)
        return 1
    return 0

def PotVal(ii, jj, kk, cc):
    global Fld, Pot, Size
    if ii < 0:
        return 30000
    if jj < 0:
        return 30000
    if ii >= Size:
        return 30000
    if jj >= Size:
        return 30000
    if Fld[ii][jj] == 0:
        return Pot[ii][jj][kk]
    if Fld[ii][jj] == -cc:
        return 30000
    return Pot[ii][jj][kk] - 30000

def SetUpd(ii, jj, cc):
    global Upd, Size
    if ii < 0:
        return
    if jj < 0:
        return
    if ii >= Size:
        return
    if jj >= Size:
        return
    Upd[ii][jj] = True

# ------------------------------
# Функции для сопоставления координат и обозначений ходов
# ------------------------------

def buildMapping():
    mapping = []
    # Верхняя половина (первая часть)
    for ii in range(Size):
        for jj in range(ii + 1):
            iCoord = ii - jj
            jCoord = jj
            token = chr(65 + jj) + str(ii - jj + 1)
            mapping.append({'i': iCoord, 'j': jCoord, 'token': token})
    # Нижняя половина (вторая часть)
    for ii in range(Size - 2, -1, -1):
        for jj in range(ii + 1):
            iCoord = Size - 1 - jj
            jCoord = Size - 1 - ii + jj
            token = chr(65 + (Size - 1 - ii + jj)) + str(Size - jj)
            mapping.append({'i': iCoord, 'j': jCoord, 'token': token})
    return mapping

def getCoordinatesFromToken(token):
    mapping = buildMapping()
    token = token.upper()
    for entry in mapping:
        if entry['token'].upper() == token:
            return {'i': entry['i'], 'j': entry['j']}
    raise Exception("Invalid move token: " + token)

def getTokenFromCoordinates(i, j):
    mapping = buildMapping()
    for entry in mapping:
        if entry['i'] == i and entry['j'] == j:
            return entry['token']
    return None

# ------------------------------
# Функция predict_next_move – симуляция игры и предсказание следующего хода
# ------------------------------

def predict_next_move(movesStr):
    """
    Принимает строку с ходами, например "A9 A9" (swap rule) или "A4 H5 G8 D3",
    симулирует игру на чистом поле (размер 11) и возвращает исходную последовательность
    с добавленным предсказанным следующим ходом.
    Механика swap rule: если на втором ходе (MoveCount==1) координата совпадает с первой,
    то это разрешается.
    """
    global IsRunning, IsBlinking, LastEvent, IsOver, Size, Fld, MoveCount, History, LastAIMove, Level, Start0, IsStart0
    # Сбрасываем состояние игры
    IsRunning = False
    IsBlinking = False
    LastEvent = ""
    IsOver = False
    SetSize(11)
    Init()
    
    moves = movesStr.strip().split()
    for k, token in enumerate(moves):
        coords = getCoordinatesFromToken(token)
        # Разрешаем повторный ход, если это второй ход (swap rule)
        if Fld[coords['i']][coords['j']] != 0:
            if not (MoveCount == 1 and token.upper() == moves[0].upper()):
                raise Exception("Обнаружен повторный ход: " + token)
        MakeMove(coords['i'], coords['j'], (k == len(moves) - 1))
        if IsOver and k < len(moves) - 1:
            raise Exception("Игра уже окончена на ходе: " + token)
    if IsOver:
        return movesStr
    currentLevel = Level[(MoveCount + (1 if Start0 else 0) + 1) % 2]
    currentCol = ((MoveCount + 1 + (1 if Start0 else 0)) % 2) * 2 - 1
    GetPot(currentLevel)
    GetBestMove(currentCol, currentLevel)
    if LastAIMove is None:
        raise Exception("Ход не предсказан")
    tokenPred = getTokenFromCoordinates(LastAIMove['i'], LastAIMove['j'])
    moves.append(tokenPred)
    return tokenPred
