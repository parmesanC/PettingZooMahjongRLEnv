# 武汉麻将所有和牌场景流程设计

## 文档信息

**创建日期**: 2026-02-04
**项目**: PettingZooRLENVMahjong
**目标**: 设计所有15个和牌场景的完整流程

---

## 流程设计模板

每个场景按照以下格式设计流程：

```
# 步骤1：玩家动作描述
    .action(player_idx, ActionType, tile)
    # → 状态转换说明

# 步骤2-3：其他玩家响应（只有一个玩家需要动作）
    .action(...)
```

---

## 场景1：硬胡自摸（已完成✅）

### 目的
验证无赖子、将牌为2/5/8的自摸硬胡

### 初始手牌
- **玩家0（庄家）**: `[0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 15]`
  - = 1万x3, 2万,3万,4万, 5万,6万,7万, 8筒x3, 9筒, 赖子(15)

### 特殊牌
- 赖子：15（2条）
- 皮子：14, 13

### 目标牌型
`111(明杠1万) 234 567 888 99(自摸9筒)`
- 无赖子（赖子已打出）
- 将牌=9筒（符合2/5/8要求）

### 游戏流程

```
# 步骤1：玩家0 赖子杠（开口）→ GONG → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_LAZY, 0)

# 步骤2：玩家0 出26（9筒）听牌 → DISCARDING → WAITING_RESPONSE → 自动跳过 → DRAWING → PLAYER_DECISION(玩家1)
    .action(0, ActionType.DISCARD, 26)

# 步骤3：玩家1 PASS → WAITING_RESPONSE.step() → 自动跳过 → DRAWING → PLAYER_DECISION(玩家2)
    .action(1, ActionType.PASS)

# 步骤4：玩家2 出0（1万）→ DISCARDING → WAITING_RESPONSE
    .action(2, ActionType.DISCARD, 0)

# 步骤5：玩家0 明杠0（1万）完成开口 → GONG → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_EXPOSED, 0)

# 步骤6：玩家0 出3（4万）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 3)

# 步骤7：玩家1 出8（9万）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(1, ActionType.DISCARD, 8)

# 步骤8：玩家2 出9 → DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(2, ActionType.DISCARD, 9)

# 步骤9：玩家3 出2 → DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(3, ActionType.DISCARD, 2)

# 步骤10：玩家0 PASS（触发自动跳过，玩家0摸牌）
    .action(0, ActionType.PASS)
    # → WAITING_RESPONSE.step() → active_responders为空 → 自动跳过
    # → DRAWING.step() → 玩家0摸25（9筒）
    # → PLAYER_DECISION

# 步骤11：玩家0 自摸硬胡 → WIN
    .action(0, ActionType.WIN, -1)
```

---

## 场景2：软胡接炮

### 目的
验证1个赖子未还原的接炮软胡

### 初始手牌
- **玩家0（庄家）**: `[0, 0, 0, 1, 2, 15, 5, 6, 7, 24, 24, 24, 26, 26]`
  - = 1万x3, 2万,3万, 赖子(15), 6万,7万, 8万, 8筒x3, 9筒x2

### 特殊牌
- 赖子：15（2条）- 在手牌中，未还原
- 皮子：14, 13

### 目标牌型
`111(赖子杠) 23(赖子)567 888 9(接炮25)`
- 1个赖子未还原（软胡）
- 接25（9筒）完成将牌

### 游戏流程

```
# 步骤1：玩家0 赖子杠（开口）→ GONG → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_LAZY, 0)
    # 杠后从牌尾摸牌

# 步骤2：玩家0 出26（9筒）听牌 → DISCARDING → WAITING_RESPONSE → 自动跳过 → DRAWING → PLAYER_DECISION(玩家1)
    .action(0, ActionType.DISCARD, 26)

# 步骤3：玩家1 PASS（触发自动跳过，玩家1摸牌）
    .action(1, ActionType.PASS)
    # → WAITING_RESPONSE.step() → 自动跳过
    # → DRAWING.step() → 玩家1摸25（9筒）
    # → PLAYER_DECISION

# 步骤4：玩家1 出25（9筒）→ DISCARDING → WAITING_RESPONSE
    .action(1, ActionType.DISCARD, 25)

# 步骤5：玩家2 PASS → WAITING_RESPONSE.step()
    .action(2, ActionType.PASS)

# 步骤6：玩家3 PASS → WAITING_RESPONSE.step() → 自动跳过 → DRAWING → PLAYER_DECISION(玩家0)
    .action(3, ActionType.PASS)
    # 注意：此时active_responders=[0]（玩家0可以胡）
    # 但玩家0需要WIN动作，所以不自动跳过

# 步骤7：玩家0 接炮软胡 → WIN
    .action(0, ActionType.WIN, -1)
```

---

## 场景3：杠上开花

### 目的
验证杠后补牌自摸胡牌（大胡）

### 初始手牌
- **玩家0（庄家）**: `[0, 0, 0, 10, 10, 13, 13, 3, 4, 5, 20, 24, 24, 26]`
  - = 1万x3, 2条x2, 4条x2, 4万,5万, 3筒, 8筒x2, 9筒

### 特殊牌
- 赖子：15（4条）
- 皮子：14, 13

### 目标牌型
`111(明杠1万) 22(明杠2条) 345 88 9(补杠开花)`
- 补杠后摸24（8筒）完成胡牌

### 游戏流程

```
# 步骤1：玩家0 出26（9筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 26)

# 步骤2：玩家1 PASS（触发自动跳过，玩家1摸牌）
    .action(1, ActionType.PASS)
    # → DRAWING.step() → 玩家1摸牌
    # → PLAYER_DECISION

# 步骤3：玩家1 出1（2万）→ DISCARDING → WAITING_RESPONSE
    .action(1, ActionType.DISCARD, 1)

# 步骤4：玩家2 PASS → WAITING_RESPONSE.step()
    .action(2, ActionType.PASS)

# 步骤5：玩家3 PASS → WAITING_RESPONSE.step()
    .action(3, ActionType.PASS)

# 步骤6：玩家0 明杠1（1万）完成开口 → GONG → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_EXPOSED, 1)
    # 杠后从牌尾摸牌

# 步骤7：玩家0 出20（3筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 20)

# 步骤8-10：玩家1,2,3 依次PASS → 自动跳过 → DRAWING → PLAYER_DECISION(玩家1)
    .action(1, ActionType.PASS)
    .action(2, ActionType.PASS)
    .action(3, ActionType.PASS)

# 步骤11：玩家1 出10（2条）→ DISCARDING → WAITING_RESPONSE
    .action(1, ActionType.DISCARD, 10)

# 步骤12：玩家2 PASS
    .action(2, ActionType.PASS)

# 步骤13：玩家3 PASS
    .action(3, ActionType.PASS)

# 步骤14：玩家0 明杠10（2条）→ GONG → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_EXPOSED, 10)
    # 杠后从牌尾摸牌

# 步骤15：玩家0 补杠10（2条）→ GONG → WAIT_ROB_KONG → 自动跳过 → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_SUPPLEMENT, 10)
    # 检查抢杠：无人能抢 → 自动跳过
    # → DRAWING_AFTER_GONG.step() → 杠上开花摸24（8筒）

# 步骤16：玩家0 杠上开花 → WIN
    .action(0, ActionType.WIN, -1)
```

---

## 场景4：全求人

### 目的
验证4面子均鸣牌得来，手牌仅剩1张，接炮胡牌（大胡）

### 初始手牌
- **玩家0（庄家）**: `[0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 24, 24, 26, 26]`
  - = 1万x2, 2万x2, 3万,4万,5万,6万,7万, 8筒x2, 9筒x2

### 特殊牌
- 赖子：15（2条）
- 皮子：14, 13

### 目标牌型
`111(明杠1万) 22(明杠2万) 345 67 8(888明杠8筒) 99(接炮)`
- 4面子全鸣牌
- 手牌剩99，接炮完成全求人

### 游戏流程

```
# 步骤1：玩家0 出26（9筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 26)

# 步骤2：玩家1 出0（1万）→ DISCARDING → WAITING_RESPONSE
    .action(1, ActionType.DISCARD, 0)

# 步骤3：玩家0 明杠0（1万）完成开口 → GONG → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_EXPOSED, 0)

# 步骤4：玩家0 出2（3万）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 2)

# 步骤5-7：玩家1,2,3 依次出牌 → 自动跳过 → DRAWING → PLAYER_DECISION(玩家0)
    .action(1, ActionType.DISCARD, 8)
    .action(2, ActionType.DISCARD, 9)
    .action(3, ActionType.DISCARD, 10)

# 步骤8：玩家0 出1（2万）→ DISCARDING → WAITING_RESPONSE
    .action(0, ActionType.DISCARD, 1)

# 步骤9：玩家2 出1（2万）→ DISCARDING → WAITING_RESPONSE
    .action(2, ActionType.DISCARD, 1)

# 步骤10：玩家0 明杠1（2万）→ GONG → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_EXPOSED, 1)

# 步骤11：玩家0 出6（7万）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 6)

# 步骤12-14：玩家1,2,3 依次出牌 → 自动跳过
    .action(1, ActionType.DISCARD, 11)
    .action(2, ActionType.DISCARD, 12)
    .action(3, ActionType.DISCARD, 13)

# 步骤15：玩家0 出24（8筒）→ DISCARDING → WAITING_RESPONSE
    .action(0, ActionType.DISCARD, 24)

# 步骤16：玩家1 出24（8筒）→ DISCARDING → WAITING_RESPONSE
    .action(1, ActionType.DISCARD, 24)

# 步骤17：玩家0 明杠24（8筒）→ GONG → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_EXPOSED, 24)

# 步骤18：玩家0 出26（9筒），手牌仅剩26（9筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 26)

# 步骤19：玩家1 出26（9筒）→ DISCARDING → WAITING_RESPONSE
    .action(1, ActionType.DISCARD, 26)

# 步骤20：玩家0 接炮全求人 → WIN
    .action(0, ActionType.WIN, -1)
```

---

## 场景5：清一色

### 目的
验证全部牌为同一花色的胡牌（大胡）

### 初始手牌
- **玩家0（庄家）**: `[0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 24, 24, 26]`
  - = 1万x3, 2万,3万,4万, 5万,6万,7万, 8万,9万, 8筒x2, 9筒

### 特殊牌
- 赖子：15（2条）
- 皮子：14, 13

### 目标牌型
`111(明杠1万) 234 567 88 99`
- 全部为万子（清一色）

### 游戏流程

```
# 步骤1：玩家0 赖子杠（开口）→ GONG → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_LAZY, 0)

# 步骤2：玩家0 出26（9筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 26)

# 步骤3-5：玩家1,2,3 依次出牌 → 自动跳过
    .action(1, ActionType.DISCARD, 25)
    .action(2, ActionType.DISCARD, 24)
    .action(3, ActionType.DISCARD, 23)

# 步骤6：玩家0 出24（8筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 24)

# 步骤7：玩家1 出0（1万）→ DISCARDING → WAITING_RESPONSE
    .action(1, ActionType.DISCARD, 0)

# 步骤8：玩家0 明杠0（1万）完成开口 → GONG → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_EXPOSED, 0)

# 步骤9：玩家0 出8（9万）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 8)

# 步骤10-12：玩家1,2,3 依次PASS → 自动跳过 → DRAWING → PLAYER_DECISION(玩家0)
    .action(1, ActionType.PASS)
    .action(2, ActionType.PASS)
    .action(3, ActionType.PASS)

# 步骤13：玩家0 摸26（9筒）清一色 → WIN
    .action(0, ActionType.WIN, -1)
```

---

## 场景6：碰碰胡

### 目的
验证面子均为刻子或杠的胡牌（大胡）

### 初始手牌
- **玩家0（庄家）**: `[0, 0, 0, 1, 1, 1, 3, 3, 3, 5, 5, 5, 24, 24]`
  - = 1万x3, 2万x3, 4万x3, 6万x3, 8筒x2

### 特殊牌
- 赖子：15（2条）
- 皮子：14, 13

### 目标牌型
`111(碰1万) 111(碰2万) 333(碰4万) 555(碰6万) 88`
- 全部刻子（碰碰胡）

### 游戏流程

```
# 步骤1：玩家0 出24（8筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 24)

# 步骤2：玩家1 出0（1万）→ DISCARDING → WAITING_RESPONSE
    .action(1, ActionType.DISCARD, 0)

# 步骤3：玩家0 碰0（1万）完成开口 → PROCESSING_MELD → PLAYER_DECISION
    .action(0, ActionType.PONG, 0)

# 步骤4：玩家0 出24（8筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 24)

# 步骤5：玩家2 出1（2万）→ DISCARDING → WAITING_RESPONSE
    .action(2, ActionType.DISCARD, 1)

# 步骤6：玩家0 碰1（2万）→ PROCESSING_MELD → PLAYER_DECISION
    .action(0, ActionType.PONG, 1)

# 步骤7：玩家0 出24（8筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 24)

# 步骤8：玩家3 出3（4万）→ DISCARDING → WAITING_RESPONSE
    .action(3, ActionType.DISCARD, 3)

# 步骤9：玩家0 碰3（4万）→ PROCESSING_MELD → PLAYER_DECISION
    .action(0, ActionType.PONG, 3)

# 步骤10：玩家0 出24（8筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 24)

# 步骤11-13：玩家1,2,3 依次PASS → 自动跳过 → DRAWING → PLAYER_DECISION(玩家0)
    .action(1, ActionType.PASS)
    .action(2, ActionType.PASS)
    .action(3, ActionType.PASS)

# 步骤14：玩家0 摸24（8筒）碰碰胡 → WIN
    .action(0, ActionType.WIN, -1)
```

---

## 场景7：风一色

### 目的
验证全部牌为风牌的胡牌（大胡）

### 初始手牌
- **玩家0（庄家）**: `[27, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33]`
  - = 东x3, 南x2, 西x2, 北x2, 中x2, 发x2, 白

### 特殊牌
- 赖子：15（2条）
- 皮子：14, 13

### 目标牌型
`东x3 南x2 西x2 北x2 中x2 发x2 白x1(接炮)`
- 全部风牌（风一色，无需组成面子和将）

### 游戏流程

```
# 步骤1：玩家0 赖子杠（开口）→ GONG → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_LAZY, 0)

# 步骤2：玩家0 出33（白板）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 33)

# 步骤3：玩家1 出33（白板）→ DISCARDING → WAITING_RESPONSE
    .action(1, ActionType.DISCARD, 33)

# 步骤4-6：玩家2,3,0 依次响应 → 自动跳过 → DRAWING → PLAYER_DECISION(玩家0)
    .action(2, ActionType.PASS)
    .action(3, ActionType.PASS)
    .action(0, ActionType.PASS)

# 步骤7：玩家0 接炮风一色 → WIN
    .action(0, ActionType.WIN, -1)
```

---

## 场景8：将一色

### 目的
验证全部牌为2、5、8的胡牌（大胡）

### 初始手牌
- **玩家0（庄家）**: `[1, 1, 1, 4, 4, 4, 7, 7, 7, 19, 19, 22, 22, 25]`
  - = 2万x3, 5万x3, 8万x3, 2筒x2, 5筒x2, 9筒

### 特殊牌
- 赖子：15（2条）
- 皮子：14, 13

### 目标牌型
`111(碰2万) 111(碰5万) 111(碰8万) 22 55(接炮5筒)`
- 全部2、5、8（将一色）

### 游戏流程

```
# 步骤1：玩家0 出25（9筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 25)

# 步骤2：玩家1 出1（2万）→ DISCARDING → WAITING_RESPONSE
    .action(1, ActionType.DISCARD, 1)

# 步骤3：玩家0 碰1（2万）完成开口 → PROCESSING_MELD → PLAYER_DECISION
    .action(0, ActionType.PONG, 1)

# 步骤4：玩家0 出25（9筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 25)

# 步骤5：玩家2 出4（5万）→ DISCARDING → WAITING_RESPONSE
    .action(2, ActionType.DISCARD, 4)

# 步骤6：玩家0 碰4（5万）→ PROCESSING_MELD → PLAYER_DECISION
    .action(0, ActionType.PONG, 4)

# 步骤7：玩家0 出25（9筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 25)

# 步骤8：玩家3 出7（8万）→ DISCARDING → WAITING_RESPONSE
    .action(3, ActionType.DISCARD, 7)

# 步骤9：玩家0 碰7（8万）→ PROCESSING_MELD → PLAYER_DECISION
    .action(0, ActionType.PONG, 7)

# 步骤10：玩家0 出25（9筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 25)

# 步骤11-13：玩家1,2,3 依次PASS → 自动跳过
    .action(1, ActionType.PASS)
    .action(2, ActionType.PASS)
    .action(3, ActionType.PASS)

# 步骤14：玩家1 出22（5筒）→ DISCARDING → WAITING_RESPONSE
    .action(1, ActionType.DISCARD, 22)

# 步骤15：玩家0 接炮将一色 → WIN
    .action(0, ActionType.WIN, -1)
```

---

## 场景9：海底捞月

### 目的
验证摸牌墙最后4张自摸胡牌（大胡）

### 初始手牌
- **玩家0（庄家）**: `[0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 15]`
  - = 1万x3, 2万,3万,4万, 5万,6万,7万, 8筒x3, 9筒, 赖子

### 特殊牌
- 赖子：15（2条）
- 皮子：14, 13

### 目标牌型
`111(明杠1万) 234 567 888 99(海底摸9筒)`
- 牌墙最后4张时摸牌胡牌

### 游戏流程

```
# 步骤1：玩家0 赖子杠（开口）→ GONG → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_LAZY, 0)

# 步骤2：玩家0 出26（9筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 26)

# 步骤3：玩家1 出0（1万）→ DISCARDING → WAITING_RESPONSE
    .action(1, ActionType.DISCARD, 0)

# 步骤4：玩家0 明杠0（1万）完成开口 → GONG → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_EXPOSED, 0)

# 步骤5-8：玩家0,1,2,3 依次出牌 → 自动跳过
    .action(0, ActionType.DISCARD, 3)
    .action(1, ActionType.DISCARD, 8)
    .action(2, ActionType.DISCARD, 9)
    .action(3, ActionType.DISCARD, 2)

# 步骤9：玩家0 PASS（触发自动跳过，摸牌）
    .action(0, ActionType.PASS)
    # → DRAWING.step() → 玩家0摸牌，牌墙剩余4张
    # → PLAYER_DECISION

# 步骤10：玩家0 海底捞月 → WIN
    .action(0, ActionType.WIN, -1)
```

---

## 场景10：抢杠和

### 目的
验证抢他人补杠牌胡牌（大胡）

### 初始手牌
- **玩家0（庄家）**: `[0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 25]`
  - = 1万x3, 2万,3万,4万, 5万,6万,7万, 8筒x3, 9筒x2

- **玩家1**: `[10, 10, 10, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]`
  - = 2条x3, 8万, 2条x3(准备补杠), 其他

### 特殊牌
- 赖子：15（2条）
- 皮子：14, 13

### 目标牌型
`111(赖子杠) 234 567 888 99(抢10杠)`
- 玩家1补杠10（2条），玩家0抢杠和

### 游戏流程

```
# 步骤1：玩家0 赖子杠（开口）→ GONG → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_LAZY, 0)

# 步骤2：玩家0 出26（9筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 26)

# 步骤3：玩家1 出10（2条）→ DISCARDING → WAITING_RESPONSE
    .action(1, ActionType.DISCARD, 10)

# 步骤4：玩家0 碰10（2条）完成开口 → PROCESSING_MELD → PLAYER_DECISION
    .action(0, ActionType.PONG, 10)

# 步骤5：玩家0 出3（4万）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 3)

# 步骤6-8：玩家1,2,3 依次出牌 → 自动跳过
    .action(1, ActionType.DISCARD, 7)
    .action(2, ActionType.DISCARD, 8)
    .action(3, ActionType.DISCARD, 9)

# 步骤9：玩家1 补杠10（2条）→ GONG → WAIT_ROB_KONG
    .action(1, ActionType.KONG_SUPPLEMENT, 10)

# 步骤10：玩家0 抢杠和 → WIN
    .action(0, ActionType.WIN, -1)
```

---

## 场景11：赖子还原硬胡

### 目的
验证赖子还原后的硬胡（赖子作为普通牌使用）

### 初始手牌
- **玩家0（庄家）**: `[0, 0, 15, 2, 3, 4, 5, 6, 7, 8, 24, 24, 24, 26]`
  - = 1万x2, 赖子(15), 3万,4万,5万,6万,7万,8万,9万, 8筒x3, 9筒

### 特殊牌
- 赖子：15（2条）- 还原为2条使用
- 皮子：14, 13

### 目标牌型
`111(1万+赖子还原2条+碰1万) 345 678 888 99`
- 赖子还原为2条，与1万、1万组成刻子
- 还原后无赖子，符合硬胡条件

### 游戏流程

```
# 步骤1：玩家0 出26（9筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 26)

# 步骤2：玩家1 出0（1万）→ DISCARDING → WAITING_RESPONSE
    .action(1, ActionType.DISCARD, 0)

# 步骤3：玩家0 碰0（1万）完成开口 → PROCESSING_MELD → PLAYER_DECISION
    .action(0, ActionType.PONG, 0)
    # melds = [Meld(0, PONG, [0,0,0])]，赖子15在手牌中

# 步骤4：玩家0 出8（9万）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 8)

# 步骤5-7：玩家1,2,3 依次PASS → 自动跳过 → DRAWING → PLAYER_DECISION(玩家0)
    .action(1, ActionType.PASS)
    .action(2, ActionType.PASS)
    .action(3, ActionType.PASS)

# 步骤8：玩家0 摸26（9筒）赖子还原硬胡 → WIN
    .action(0, ActionType.WIN, -1)
    # 胡牌检测时：
    # - 手牌 = [15, 2, 3, 4, 5, 6, 7, 24, 24, 24, 26, 26]
    # - melds = [0碰]
    # - 赖子15还原为2条（10），与2,3组成顺子345(万子0123)
    # - 实际牌型 = 碰(0,0,0) 顺子(0,1,2) 顺子(3,4,5) 刻子(24,24,24) 将(26,26)
    # - 无赖子，将牌26（9筒），符合硬胡条件
```

---

## 场景12：边界测试 - 最小起胡番数

### 目的
验证乘积刚好16（起胡线）的边界情况

### 初始手牌
- **玩家0（庄家）**: `[0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 15]`
  - = 1万x3, 2万,3万,4万, 5万,6万,7万, 8筒x3, 9筒, 赖子

### 特殊牌
- 赖子：15（2条）
- 皮子：14, 13

### 目标牌型
`111(明杠1万) 234 567 888 99`
- 硬胡(x2) x 自摸(x2) x 明杠x2 = 16（刚好起胡）

### 游戏流程

```
# 步骤1：玩家0 赖子杠（开口）→ GONG → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_LAZY, 0)

# 步骤2：玩家0 出26（9筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 26)

# 步骤3：玩家1 出0（1万）→ DISCARDING → WAITING_RESPONSE
    .action(1, ActionType.DISCARD, 0)

# 步骤4：玩家0 明杠0（1万）完成开口 → GONG → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_EXPOSED, 0)

# 步骤5-8：玩家0,1,2,3 依次出牌 → 自动跳过
    .action(0, ActionType.DISCARD, 3)
    .action(1, ActionType.DISCARD, 8)
    .action(2, ActionType.DISCARD, 9)
    .action(3, ActionType.DISCARD, 2)

# 步骤9：玩家0 PASS → 自动跳过 → DRAWING → PLAYER_DECISION
    .action(0, ActionType.PASS)

# 步骤10：玩家0 自摸硬胡 → WIN
    .action(0, ActionType.WIN, -1)
    # 番数计算：硬胡x2 x 自摸x2 x 明杠x2 x 赖子杠x4 = 64 > 16 ✓
```

---

## 场景13：边界测试 - 赖子数量限制

### 目的
验证小胡赖子数量>1时不能胡牌

### 初始手牌
- **玩家0（庄家）**: `[0, 0, 15, 15, 2, 3, 4, 5, 6, 7, 8, 24, 24, 24]`
  - = 1万x2, 赖子x2, 3万,4万,5万,6万,7万,8万,9万, 8筒x3

### 特殊牌
- 赖子：15（2条）
- 皮子：14, 13

### 预期结果
- 手牌有2个赖子，尝试胡牌时应该失败
- 返回不能胡牌的错误

### 游戏流程

```
# 步骤1：玩家0 出24（8筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 24)

# 步骤2：玩家1 出0（1万）→ DISCARDING → WAITING_RESPONSE
    .action(1, ActionType.DISCARD, 0)

# 步骤3：玩家0 碰0（1万）完成开口 → PROCESSING_MELD → PLAYER_DECISION
    .action(0, ActionType.PONG, 0)

# 步骤4：玩家0 出24（8筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 24)

# 步骤5-7：玩家1,2,3 依次PASS → 自动跳过 → DRAWING → PLAYER_DECISION
    .action(1, ActionType.PASS)
    .action(2, ActionType.PASS)
    .action(3, ActionType.PASS)

# 步骤8：玩家0 摸24（8筒）尝试胡牌 → 失败
    .action(0, ActionType.WIN, -1)
    # 预期：ValueError - 赖子数量超过限制（小胡最多1个）
```

---

## 场景14：边界测试 - 未开口不能胡牌

### 目的
验证未开口时不能胡牌

### 初始手牌
- **玩家0（庄家）**: `[0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 26]`
  - = 1万x3, 2万,3万,4万, 5万,6万,7万, 8筒x3, 9筒, 9筒

### 特殊牌
- 赖子：15（2条）
- 皮子：14, 13

### 预期结果
- 未完成开口，尝试胡牌时应该失败

### 游戏流程

```
# 步骤1：玩家0 出26（9筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 26)

# 步骤2-4：玩家1,2,3 依次PASS → 自动跳过 → DRAWING → PLAYER_DECISION
    .action(1, ActionType.PASS)
    .action(2, ActionType.PASS)
    .action(3, ActionType.PASS)

# 步骤3：玩家0 摸26（9筒）尝试胡牌 → 失败
    .action(0, ActionType.WIN, -1)
    # 预期：ValueError - 未开口不能胡牌
```

---

## 场景15：边界测试 - 皮/红中不能胡牌

### 目的
验证胡牌时手牌中不能有皮或红中

### 初始手牌
- **玩家0（庄家）**: `[0, 0, 0, 1, 2, 3, 4, 5, 6, 24, 24, 24, 25, 31]`
  - = 1万x3, 2万,3万,4万, 5万,6万,7万, 8筒x3, 9筒, 红中(31)

### 特殊牌
- 赖子：15（2条）
- 皮子：14, 13

### 预期结果
- 手牌有红中，尝试胡牌时应该失败

### 游戏流程

```
# 步骤1：玩家0 赖子杠（开口）→ GONG → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_LAZY, 0)

# 步骤2：玩家0 出25（9筒）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 25)

# 步骤3：玩家1 出0（1万）→ DISCARDING → WAITING_RESPONSE
    .action(1, ActionType.DISCARD, 0)

# 步骤4：玩家0 明杠0（1万）完成开口 → GONG → DRAWING_AFTER_GONG → PLAYER_DECISION
    .action(0, ActionType.KONG_EXPOSED, 0)

# 步骤5：玩家0 出31（红中）→ DISCARDING → WAITING_RESPONSE → 自动跳过
    .action(0, ActionType.DISCARD, 31)

# 步骤6-8：玩家1,2,3 依次PASS → 自动跳过 → DRAWING → PLAYER_DECISION
    .action(1, ActionType.PASS)
    .action(2, ActionType.PASS)
    .action(3, ActionType.PASS)

# 步骤9：玩家0 摸25（9筒）尝试胡牌 → 失败
    .action(0, ActionType.WIN, -1)
    # 预期：ValueError - 弃牌堆有红中，不能胡牌
    # 或者：如果红中未被打出，手牌有红中不能胡牌
```

---

## 总结

| 场景 | 类型 | 关键点 |
|------|------|--------|
| 1 | 硬胡自摸 | 无赖子，将牌2/5/8 |
| 2 | 软胡接炮 | 1个赖子未还原 |
| 3 | 杠上开花 | 补杠后自摸 |
| 4 | 全求人 | 4面子全鸣牌，手牌剩1张 |
| 5 | 清一色 | 全部同花色 |
| 6 | 碰碰胡 | 全部刻子/杠 |
| 7 | 风一色 | 全部风牌 |
| 8 | 将一色 | 全部2/5/8 |
| 9 | 海底捞月 | 牌墙最后4张自摸 |
| 10 | 抢杠和 | 抢补杠牌 |
| 11 | 赖子还原硬胡 | 赖子作普通牌使用 |
| 12 | 边界-最小起胡 | 乘积=16 |
| 13 | 边界-赖子超限 | 小胡>1个赖子 |
| 14 | 边界-未开口 | 未开口不能胡 |
| 15 | 边界-红中/皮 | 有红中/皮不能胡 |

---

**文档版本**: 1.0
**最后更新**: 2026-02-04
**作者**: Claude Code
