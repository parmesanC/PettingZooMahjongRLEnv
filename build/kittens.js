const SCI_TH = 0.95;          // 科学使用率阈值（95%触发）
const INT_MS = 8000;          // 检查间隔（加快响应速度）

let compCounter = 0;          // 概要合成计数器（内存级）

const autoCraft = setInterval(() => {
    const sci = game.resPool.get("science");
    const manuscript = game.resPool.get("manuscript");
    const compediumCount = game.workshop.getCraftAllCount("compedium");
    const blueprintCount = game.workshop.getCraftAllCount("blueprint");

    // 获取配方信息
    const compediumRecipe = game.workshop.getCraft("compedium");
    const blueprintRecipe = game.workshop.getCraft("blueprint");
    const compPrices = game.workshop.getCraftPrice(compediumRecipe);
    const bpPrices = game.workshop.getCraftPrice(blueprintRecipe);

    // 提取资源需求
    const manuscriptCost = compPrices.find(p => p.name === "manuscript").val;
    const compediumValue = game.resPool.get("compedium").value;

    // 条件1: 科学资源基本满额
    if (sci.value < sci.maxValue * SCI_TH) return;

    // 计算最大可合成次数
    const maxByScience = compediumCount;
    const maxByManuscript = Math.floor((manuscript.value - compediumValue) / manuscriptCost);
    const maxSafeCraft = Math.min(maxByScience, maxByManuscript); // 安全限制

    if (maxSafeCraft <= 0) return;

    // 优先合成概要
    if (compCounter < 6 && maxSafeCraft > 0) {
        const craftCount = maxSafeCraft;
        game.workshop.craft("compedium", craftCount, true, true, false);
        compCounter += 1;
        console.log(`[${new Date().toLocaleTimeString()}] 合成概要 ×${craftCount} | 累计 ${compCounter}`);
    }
    // 每6次概要合成1次蓝图
    else if (compCounter >= 6) {
        const bpCraftCount = blueprintCount;
        if (bpCraftCount > 0) {
            game.workshop.craft("blueprint", bpCraftCount, true, true, false);
            compCounter = 0;
            console.log(`[${new Date().toLocaleTimeString()}] 合成蓝图 ×${bpCraftCount} | 剩余累计 ${compCounter}`);
        }
    }
}, INT_MS);

// 停止函数
window.stopAutoCraft = () => clearInterval(autoCraft);


const THRESHOLD = 0.95;          // 95% 触发
const BATCH_MS  = 2000;          // 每 2 秒检查一次

const autoCraftManuscript = setInterval(() => {
    const c = game.resPool.get("culture");
    if (c.value >= c.maxValue * THRESHOLD) {
        // 1. 计算当前最多能合成多少次手稿
        const maxTimes = game.workshop.getCraftAllCount("manuscript");
        if (maxTimes > 0) {
            // 2. 一次全部合成
            game.workshop.craft("manuscript", maxTimes, true, true, false);
            console.log(
                `[${new Date().toLocaleTimeString()}] 文化 ${c.value.toFixed(1)}/${c.maxValue} ` +
                `→ 已自动合成 ${maxTimes} 份手稿`
            );
        } else {
            console.log("文化已满，但暂无可合成手稿次数。");
        }
    }
}, BATCH_MS);


[
    {
        "name": "wood",
        "label": "精炼猫薄荷",
        "description": "坚固的猫薄荷木块。加工起来很麻烦，却是很好的建筑材料",
        "prices": [
            {
                "name": "catnip",
                "val": 50
            }
        ],
        "ignoreBonuses": true,
        "progressHandicap": 1,
        "tier": 1,
        "unlocked": true,
        "value": 0,
        "progress": 0,
        "isLimited": false
    },
    {
        "name": "beam",
        "label": "木梁",
        "description": "简单的木质支撑结构，是建造高级建筑的基础",
        "prices": [
            {
                "name": "wood",
                "val": 175
            }
        ],
        "progressHandicap": 1,
        "tier": 1,
        "unlocked": true,
        "value": 0,
        "progress": 0,
        "isLimited": false
    },
    {
        "name": "slab",
        "label": "石板",
        "description": "由矿物组成的石板，是建造高级建筑的基础",
        "prices": [
            {
                "name": "minerals",
                "val": 250
            }
        ],
        "progressHandicap": 1,
        "tier": 1,
        "unlocked": true,
        "value": 0,
        "progress": 0,
        "isLimited": false
    },
    {
        "name": "plate",
        "label": "金属板",
        "description": "一块金属板，是建造高级建筑的基础",
        "prices": [
            {
                "name": "iron",
                "val": 125
            }
        ],
        "progressHandicap": 4,
        "tier": 1,
        "unlocked": true,
        "value": 0,
        "progress": 0,
        "isLimited": false
    },
    {
        "name": "steel",
        "label": "钢",
        "description": "用铁和煤冶炼得到的坚实金属，用于制作齿轮和复杂的机械",
        "prices": [
            {
                "name": "coal",
                "val": 100
            },
            {
                "name": "iron",
                "val": 100
            }
        ],
        "progressHandicap": 4,
        "tier": 2,
        "unlocked": true,
        "value": 0,
        "progress": 0,
        "isLimited": false
    },
    {
        "name": "concrate",
        "label": "混凝土",
        "description": "一块钢筋混凝土",
        "prices": [
            {
                "name": "slab",
                "val": 2500
            },
            {
                "name": "steel",
                "val": 25
            }
        ],
        "progressHandicap": 9,
        "tier": 4,
        "unlocked": true,
        "value": 0,
        "progress": 0,
        "isLimited": false
    },
    {
        "name": "gear",
        "label": "齿轮",
        "description": "自动化结构中不可或缺的组成部分",
        "prices": [
            {
                "name": "steel",
                "val": 15
            }
        ],
        "progressHandicap": 5,
        "tier": 3,
        "unlocked": true,
        "value": 0,
        "progress": 0,
        "isLimited": false
    },
    {
        "name": "alloy",
        "label": "合金",
        "description": "钢铁和钛制成的坚实合金，用于建造高级建筑和工坊升级",
        "prices": [
            {
                "name": "titanium",
                "val": 10
            },
            {
                "name": "steel",
                "val": 75
            }
        ],
        "progressHandicap": 7,
        "tier": 4,
        "unlocked": true,
        "value": 0,
        "progress": 0,
        "isLimited": false
    },
    {
        "name": "eludium",
        "label": "E合金",
        "description": "难得素与钛制成的极为罕见珍贵的合金",
        "prices": [
            {
                "name": "unobtainium",
                "val": 1000
            },
            {
                "name": "alloy",
                "val": 2500
            }
        ],
        "progressHandicap": 300,
        "tier": 5,
        "unlocked": false,
        "value": 0,
        "progress": 0,
        "isLimited": true
    },
    {
        "name": "scaffold",
        "label": "脚手架",
        "description": "木梁制作的大型结构，用于建造非常复杂的建筑和东西",
        "prices": [
            {
                "name": "beam",
                "val": 50
            }
        ],
        "progressHandicap": 2,
        "tier": 2,
        "unlocked": true,
        "value": 0,
        "progress": 0,
        "isLimited": false
    },
    {
        "name": "ship",
        "label": "贸易船",
        "description": "船可以用于发现新的文明。可以提高获得某种资源的机会",
        "prices": [
            {
                "name": "starchart",
                "val": 25
            },
            {
                "name": "plate",
                "val": 150
            },
            {
                "name": "scaffold",
                "val": 100
            }
        ],
        "upgrades": {
            "buildings": [
                "harbor"
            ]
        },
        "progressHandicap": 20,
        "tier": 3,
        "unlocked": true,
        "value": 1,
        "progress": 0.011481326994799805,
        "isLimited": false,
        "isLimitedAmt": false
    },
    {
        "name": "tanker",
        "label": "油轮",
        "description": "将石油的存储上限提高 500 点",
        "prices": [
            {
                "name": "alloy",
                "val": 1250
            },
            {
                "name": "ship",
                "val": 200
            },
            {
                "name": "blueprint",
                "val": 5
            }
        ],
        "upgrades": {
            "buildings": [
                "harbor"
            ]
        },
        "progressHandicap": 20,
        "tier": 5,
        "unlocked": true,
        "value": 0,
        "progress": 0,
        "isLimited": false
    },
    {
        "name": "kerosene",
        "label": "煤油",
        "description": "加工石油得到的火箭燃料",
        "prices": [
            {
                "name": "oil",
                "val": 7500
            }
        ],
        "progressHandicap": 5,
        "tier": 2,
        "unlocked": true,
        "value": 0,
        "progress": 0,
        "isLimited": false
    },
    {
        "name": "parchment",
        "label": "羊皮纸",
        "description": "使用动物毛皮制作的用于书写的材料，是文化建筑的基础",
        "prices": [
            {
                "name": "furs",
                "val": 175
            }
        ],
        "progressHandicap": 1,
        "tier": 1,
        "unlocked": true,
        "value": 0,
        "progress": 0,
        "isLimited": false
    },
    {
        "name": "manuscript",
        "label": "手稿",
        "description": "科技发展所需的书面文件。每份手稿会轻微增加最大文化上限（这一效果的增益递减）",
        "prices": [
            {
                "name": "culture",
                "val": 400
            },
            {
                "name": "parchment",
                "val": 25
            }
        ],
        "progressHandicap": 2,
        "tier": 2,
        "unlocked": true,
        "value": 1,
        "progress": 0.8072332469709286,
        "isLimited": false,
        "isLimitedAmt": false
    },
    {
        "name": "compedium",
        "label": "概要",
        "description": "猫类所有现代知识的总和。每份概要会使最大科学上限 +10（这一效果无法超过建筑的最大科学上限）",
        "prices": [
            {
                "name": "science",
                "val": 10000
            },
            {
                "name": "manuscript",
                "val": 50
            }
        ],
        "progressHandicap": 5,
        "tier": 3,
        "unlocked": true,
        "value": 0,
        "progress": 0,
        "isLimited": false
    },
    {
        "name": "blueprint",
        "label": "蓝图",
        "description": "有着蓝色线条的奇怪的纸",
        "prices": [
            {
                "name": "science",
                "val": 25000
            },
            {
                "name": "compedium",
                "val": 25
            }
        ],
        "progressHandicap": 10,
        "tier": 3,
        "unlocked": true,
        "value": 0,
        "progress": 0,
        "isLimited": false
    },
    {
        "name": "thorium",
        "label": "钍",
        "description": "具有高放射性且不稳定的燃料",
        "prices": [
            {
                "name": "uranium",
                "val": 250
            }
        ],
        "progressHandicap": 5,
        "tier": 3,
        "unlocked": false,
        "value": 0,
        "progress": 0,
        "isLimited": false
    },
    {
        "name": "megalith",
        "label": "巨石",
        "description": "用于建造巨型结构的大石块",
        "prices": [
            {
                "name": "beam",
                "val": 25
            },
            {
                "name": "slab",
                "val": 50
            },
            {
                "name": "plate",
                "val": 5
            }
        ],
        "progressHandicap": 5,
        "tier": 3,
        "unlocked": true,
        "value": 0,
        "progress": 0,
        "isLimited": false
    },
    {
        "name": "bloodstone",
        "label": "血石",
        "description": "一件奇怪的珠宝，据说是由时间和上古神的鲜血制成的",
        "prices": [
            {
                "name": "timeCrystal",
                "val": 5000
            },
            {
                "name": "relic",
                "val": 10000
            }
        ],
        "progressHandicap": 7500,
        "tier": 5,
        "unlocked": false,
        "value": 0,
        "progress": 0,
        "isLimited": false
    },
    {
        "name": "tMythril",
        "label": "T秘银",
        "description": "T秘银 (待定)",
        "prices": [
            {
                "name": "bloodstone",
                "val": 5
            },
            {
                "name": "ivory",
                "val": 1000
            },
            {
                "name": "titanium",
                "val": 500
            }
        ],
        "progressHandicap": 10000,
        "tier": 7,
        "unlocked": false,
        "value": 0,
        "progress": 0,
        "isLimited": false
    }
]


const THRESHOLD = 0.95;          // 95% 就触发，想 100% 改成 1

const autoPraise = setInterval(() => {
    const f = game.resPool.get("faith");
    if (f.value >= f.maxValue * THRESHOLD) {
        game.praise({ preventDefault: () => {} });
        console.log(`[${new Date().toLocaleTimeString()}] 信仰 ${f.value.toFixed(1)}/${f.maxValue} → 已赞美`);
    }
}, 2000);

const BATCH_MS  = 3000;          // 每 3 秒检查一次

const autoCraftManuscript = setInterval(() => {
    const c = game.resPool.get("culture");
    if (c.value >= c.maxValue * THRESHOLD) {
        // 1. 计算当前最多能合成多少次手稿
        const maxTimes = game.workshop.getCraftAllCount("manuscript");
        if (maxTimes > 0) {
            // 2. 一次全部合成
            game.workshop.craft("manuscript", maxTimes, true, true, false);
            console.log(
                `[${new Date().toLocaleTimeString()}] 文化 ${c.value.toFixed(1)}/${c.maxValue} ` +
                `→ 已自动合成 ${maxTimes} 份手稿`
            );
        } else {
            console.log("文化已满，但暂无可合成手稿次数。");
        }
    }
}, BATCH_MS);

const autoHuntAll = setInterval(() => {
    const mp = game.resPool.get("manpower");
    if (mp.value >= mp.maxValue * THRESHOLD) {
        game.village.huntAll();  // 一键派出所有猎人
        console.log(
            `[${new Date().toLocaleTimeString()}] 喵力 ${mp.value.toFixed(1)}/${mp.maxValue} ` +
            `→ 已自动执行 huntAll`
        );
    }
}, 100);

const SCI_TH = 0.95;          // 科学使用率阈值（95%触发）
const INT_MS = 8000;          // 检查间隔（加快响应速度）

let compCounter = 0;          // 概要合成计数器（内存级）

const autoCraft = setInterval(() => {
    const sci = game.resPool.get("science");
    const manuscript = game.resPool.get("manuscript");
    const compediumCount = game.workshop.getCraftAllCount("compedium");
    const blueprintCount = game.workshop.getCraftAllCount("blueprint");

    // 获取配方信息
    const compediumRecipe = game.workshop.getCraft("compedium");
    const blueprintRecipe = game.workshop.getCraft("blueprint");
    const compPrices = game.workshop.getCraftPrice(compediumRecipe);
    const bpPrices = game.workshop.getCraftPrice(blueprintRecipe);

    // 提取资源需求
    const manuscriptCost = compPrices.find(p => p.name === "manuscript").val;
    const compediumValue = game.resPool.get("compedium").value;

    // 条件1: 科学资源基本满额
    if (sci.value < sci.maxValue * SCI_TH) return;

    // 计算最大可合成次数
    const maxByScience = compediumCount;
    const maxByManuscript = Math.floor((manuscript.value - compediumValue) / manuscriptCost);
    const maxSafeCraft = Math.min(maxByScience, maxByManuscript); // 安全限制

    if (maxSafeCraft <= 0) return;

    // 优先合成概要
    if (compCounter < 6 && maxSafeCraft > 0) {
        const craftCount = maxSafeCraft;
        game.workshop.craft("compedium", craftCount, true, true, false);
        compCounter += 1;
        console.log(`[${new Date().toLocaleTimeString()}] 合成概要 ×${craftCount} | 累计 ${compCounter}`);
    }
    // 每6次概要合成1次蓝图
    else if (compCounter >= 6) {
        const bpCraftCount = blueprintCount;
        if (bpCraftCount > 0) {
            game.workshop.craft("blueprint", bpCraftCount, true, true, false);
            compCounter = 0;
            console.log(`[${new Date().toLocaleTimeString()}] 合成蓝图 ×${bpCraftCount} | 剩余累计 ${compCounter}`);
        }
    }
}, INT_MS);

const autoCraftKerosene = setInterval(() => {
    const c = game.resPool.get("oil");
    if (c.value >= c.maxValue * THRESHOLD) {
        const maxTimes = game.workshop.getCraftAllCount("kerosene");
        if (maxTimes > 0) {
            game.workshop.craft("kerosene", maxTimes, true, true, false);
            console.log(
                `[${new Date().toLocaleTimeString()}] 石油 ${c.value.toFixed(1)}/${c.maxValue} ` +
                `→ 已自动合成 ${maxTimes} 份煤油`
            );
        } else {
            console.log("石油已满，但暂无可合成煤油次数。");
        }
    }
}, 15000);

const autoCraftEludium = setInterval(() => {
    const c = game.resPool.get("unobtainium");
    if (c.value >= c.maxValue * THRESHOLD) {
        const maxTimes = game.workshop.getCraftAllCount("eludium");
        if (maxTimes > 0) {
            game.workshop.craft("eludium", maxTimes, true, true, false);
            console.log(
                `[${new Date().toLocaleTimeString()}] 难得素 ${c.value.toFixed(1)}/${c.maxValue} ` +
                `→ 已自动合成 ${maxTimes} 份E合金`
            );
        } else {
            console.log("难得素已满，但暂无可合成e合金次数。");
        }
    }
}, 5000);




const autoCraftSteel = setInterval(() => {
    const c = game.resPool.get("coal");
    if (c.value >= c.maxValue * THRESHOLD) {
        const maxTimes = game.workshop.getCraftAllCount("steel");
        if (maxTimes > 0) {
            game.workshop.craft("steel", maxTimes, true, true, false);
            console.log(
                `[${new Date().toLocaleTimeString()}] 煤炭 ${c.value.toFixed(1)}/${c.maxValue} ` +
                `→ 已自动合成 ${maxTimes} 份钢`
            );
            const plateMaxTimes = game.workshop.getCraftAllCount("plate");
            if (plateMaxTimes > 0) {
                game.workshop.craft("plate", plateMaxTimes, true, true, false);
                console.log(
                    `[${new Date().toLocaleTimeString()}] 煤炭 ${c.value.toFixed(1)}/${c.maxValue} ` +
                    `→ 已自动合成 ${plateMaxTimes} 份板`
                );
            }
        } else {
            console.log("煤炭已满，但暂无可合成钢次数。");
        }
    }
}, 500);


const autoCraftThorium = setInterval(() => {
    const c = game.resPool.get("uranium");
    if (c.value >= c.maxValue * THRESHOLD) {
        const maxTimes = game.workshop.getCraftAllCount("thorium");
        if (maxTimes > 0) {
            game.workshop.craft("thorium", maxTimes, true, true, false);
            console.log(
                `[${new Date().toLocaleTimeString()}] 铀 ${c.value.toFixed(1)}/${c.maxValue} ` +
                `→ 已自动合成 ${maxTimes} 份钍`
            );
        } else {
            console.log("铀已满，但暂无可合成钍次数。");
        }
    }
}, 500);

const autoCraftBeam = setInterval(() => {
    const c = game.resPool.get("wood");
    if (c.value >= c.maxValue * THRESHOLD) {
        const maxTimes = game.workshop.getCraftAllCount("beam");
        if (maxTimes > 0) {
            game.workshop.craft("beam", maxTimes, true, true, false);
            console.log(
                `[${new Date().toLocaleTimeString()}] 木材 ${c.value.toFixed(1)}/${c.maxValue} ` +
                `→ 已自动合成 ${maxTimes} 份梁`
            );
        } else {
            console.log("木材已满，但暂无可合成梁次数。");
        }
    }
}, 500);


const autoCraftSlab = setInterval(() => {
    const c = game.resPool.get("minerals");
    if (c.value >= c.maxValue * THRESHOLD) {
        const maxTimes = game.workshop.getCraftAllCount("slab");
        if (maxTimes > 0) {
            game.workshop.craft("slab", maxTimes, true, true, false);
            console.log(
                `[${new Date().toLocaleTimeString()}] 矿物 ${c.value.toFixed(1)}/${c.maxValue} ` +
                `→ 已自动合成 ${maxTimes} 份板`
            );
        } else {
            console.log("矿物已满，但暂无可合成板次数。");
        }
    }
}, 500);