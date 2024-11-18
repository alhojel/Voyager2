const fs = require("fs");
const express = require("express");
const bodyParser = require("body-parser");
const mineflayer = require("mineflayer");
const { mineflayer: mineflayerViewer } = require('prismarine-viewer')

const skills = require("./lib/skillLoader");
const { initCounter, getNextTime } = require("./lib/utils");
const obs = require("./lib/observation/base");
const OnChat = require("./lib/observation/onChat");
const OnError = require("./lib/observation/onError");
const { Voxels, BlockRecords } = require("./lib/observation/voxels");
const Status = require("./lib/observation/status");
const Inventory = require("./lib/observation/inventory");
const OnSave = require("./lib/observation/onSave");
const Chests = require("./lib/observation/chests");
const { plugin: tool } = require("mineflayer-tool");

let bot = null;

const app = express();

app.use(bodyParser.json({ limit: "50mb" }));
app.use(bodyParser.urlencoded({ limit: "50mb", extended: false }));

app.post("/start", (req, res) => {
    if (bot) onDisconnect("Restarting bot");
    bot = null;
    console.log(req.body);
    bot = mineflayer.createBot({
        host: "localhost", // minecraft server ip
        port: req.body.port, // minecraft server port
        //username: "bot",
        username: req.body.username || "bot",
        disableChatSigning: true,
        checkTimeoutInterval: 60 * 60 * 1000,
    });
    bot.once("error", onConnectionFailed);

    // Event subscriptions
    bot.waitTicks = req.body.waitTicks;
    bot.globalTickCounter = 0;
    bot.stuckTickCounter = 0;
    bot.stuckPosList = [];
    bot.iron_pickaxe = false;

    bot.on("kicked", onDisconnect);

    // mounting will cause physicsTick to stop
    bot.on("mount", () => {
        bot.dismount();
    });

    bot.once("spawn", async () => {
        bot.removeListener("error", onConnectionFailed);
        let itemTicks = 1;
        if (req.body.reset === "hard") {
            bot.chat("/clear @s");
            bot.chat("/kill @s");
            const inventory = req.body.inventory ? req.body.inventory : {};
            const equipment = req.body.equipment
                ? req.body.equipment
                : [null, null, null, null, null, null];
            for (let key in inventory) {
                bot.chat(`/give @s minecraft:${key} ${inventory[key]}`);
                itemTicks += 1;
            }
            const equipmentNames = [
                "armor.head",
                "armor.chest",
                "armor.legs",
                "armor.feet",
                "weapon.mainhand",
                "weapon.offhand",
            ];
            for (let i = 0; i < 6; i++) {
                if (i === 4) continue;
                if (equipment[i]) {
                    bot.chat(
                        `/item replace entity @s ${equipmentNames[i]} with minecraft:${equipment[i]}`
                    );
                    itemTicks += 1;
                }
            }
        }

        if (req.body.position) {
            bot.chat(
                `/tp @s ${req.body.position.x} ${req.body.position.y} ${req.body.position.z}`
            );
        }

        // if iron_pickaxe is in bot's inventory
        if (
            bot.inventory.items().find((item) => item.name === "iron_pickaxe")
        ) {
            bot.iron_pickaxe = true;
        }

        const { pathfinder } = require("mineflayer-pathfinder");
        const tool = require("mineflayer-tool").plugin;
        const collectBlock = require("mineflayer-collectblock").plugin;
        const pvp = require("mineflayer-pvp").plugin;
        //const minecraftHawkEye = require("minecrafthawkeye");
        bot.loadPlugin(pathfinder);
        bot.loadPlugin(tool);
        bot.loadPlugin(collectBlock);
        bot.loadPlugin(pvp);
        //bot.loadPlugin(minecraftHawkEye);

        try {
            console.log('Bot spawned, initializing viewer...');
            mineflayerViewer(bot, { 
                port: req.body.viewerPort || 3001,  // Use dynamic viewer port
                firstPerson: true,
                viewDistance: 6
            });
        } catch (error) {
            console.error('Error initializing viewer:', error);
        }
        console.log('Starting vision capture after delay...');
        // Ensure the logging folder exists
        const loggingFolder = path.resolve('/Users/daisysong/Desktop/CS194agent/Voyager_OAI/logs/visions');
        if (!fs.existsSync(loggingFolder)) {
            fs.mkdirSync(loggingFolder, { recursive: true });
        }
        //setupVisionCapture(bot);
         //Wait for a short duration to ensure viewer is ready
        setTimeout(() => {
            console.log('Starting vision capture after delay...');
            setupVisionCapture(bot);
        }, 2000); // Delay in milliseconds (adjust as needed)
         // new add for camera
        // Camera synchronization logic
        // setInterval(() => {
        //     const cameraOffset = new Vec3(0, 1.6, 0); // Adjust Y value for height
        //     const cameraPosition = bot.entity.position.plus(cameraOffset);
        //     // Assuming you have a way to set the camera position in the viewer
        //     if (bot.viewer) {
        //         bot.viewer.setCameraPosition(cameraPosition);
        //         bot.viewer.setCameraYaw(bot.entity.yaw);
        //         bot.viewer.setCameraPitch(bot.entity.pitch);
        //     }
        // }, 50); // Update every 50ms (adjust as needed)
        // new add for camera

        // Ensure bot.viewer is defined before using it
        // if (bot.viewer) {
        //     console.log('Viewer initialized, starting vision capture...');
        //     setupVisionCapture(bot);
        // } else {
        //     console.error('bot.viewer is undefined. Viewer may not have initialized correctly.');
        // }
        // Check if 'viewerReady' or similar event exists
        // bot.viewer.once('viewerReady', () => {
        //     console.log('Viewer is ready, starting vision capture...');
        //     setupVisionCapture(bot);
        // });

        bot.collectBlock.movements.digCost = 0;
        bot.collectBlock.movements.placeCost = 0;

        obs.inject(bot, [
            OnChat,
            OnError,
            Voxels,
            Status,
            Inventory,
            OnSave,
            Chests,
            BlockRecords,
        ]);
        skills.inject(bot);

        if (req.body.spread) {
            bot.chat(`/spreadplayers ~ ~ 0 300 under 80 false @s`);
            await bot.waitForTicks(bot.waitTicks);
        }

        await bot.waitForTicks(bot.waitTicks * itemTicks);
        res.json(bot.observe());

        initCounter(bot);
        bot.chat("/gamerule keepInventory true");
        bot.chat("/gamerule doDaylightCycle false");
    });

    function onConnectionFailed(e) {
        console.log(e);
        bot = null;
        res.status(400).json({ error: e });
    }
    function onDisconnect(message) {
        if (bot.viewer) {
            bot.viewer.close();
        }
        bot.end();
        console.log(message);
        bot = null;
    }
});

app.post("/step", async (req, res) => {
    // import useful package
    let response_sent = false;
    function otherError(err) {
        console.log("Uncaught Error");
        bot.emit("error", handleError(err));
        bot.waitForTicks(bot.waitTicks).then(() => {
            if (!response_sent) {
                response_sent = true;
                res.json(bot.observe());
            }
        });
    }

    process.on("uncaughtException", otherError);

    const mcData = require("minecraft-data")(bot.version);
    mcData.itemsByName["leather_cap"] = mcData.itemsByName["leather_helmet"];
    mcData.itemsByName["leather_tunic"] =
        mcData.itemsByName["leather_chestplate"];
    mcData.itemsByName["leather_pants"] =
        mcData.itemsByName["leather_leggings"];
    mcData.itemsByName["leather_boots"] = mcData.itemsByName["leather_boots"];
    mcData.itemsByName["lapis_lazuli_ore"] = mcData.itemsByName["lapis_ore"];
    mcData.blocksByName["lapis_lazuli_ore"] = mcData.blocksByName["lapis_ore"];
    const {
        Movements,
        goals: {
            Goal,
            GoalBlock,
            GoalNear,
            GoalXZ,
            GoalNearXZ,
            GoalY,
            GoalGetToBlock,
            GoalLookAtBlock,
            GoalBreakBlock,
            GoalCompositeAny,
            GoalCompositeAll,
            GoalInvert,
            GoalFollow,
            GoalPlaceBlock,
        },
        pathfinder,
        Move,
        ComputedPath,
        PartiallyComputedPath,
        XZCoordinates,
        XYZCoordinates,
        SafeBlock,
        GoalPlaceBlockOptions,
    } = require("mineflayer-pathfinder");
    const { Vec3 } = require("vec3");

    // Set up pathfinder
    const movements = new Movements(bot, mcData);
    bot.pathfinder.setMovements(movements);

    bot.globalTickCounter = 0;
    bot.stuckTickCounter = 0;
    bot.stuckPosList = [];

    function onTick() {
        bot.globalTickCounter++;
        if (bot.pathfinder.isMoving()) {
            bot.stuckTickCounter++;
            if (bot.stuckTickCounter >= 100) {
                onStuck(1.5);
                bot.stuckTickCounter = 0;
            }
        }
    }

    bot.on("physicTick", onTick);

    // initialize fail count
    let _craftItemFailCount = 0;
    let _killMobFailCount = 0;
    let _mineBlockFailCount = 0;
    let _placeItemFailCount = 0;
    let _smeltItemFailCount = 0;

    // Retrieve array form post bod
    const code = req.body.code;
    const programs = req.body.programs;
    bot.cumulativeObs = [];
    await bot.waitForTicks(bot.waitTicks);
    const r = await evaluateCode(code, programs);
    process.off("uncaughtException", otherError);
    if (r !== "success") {
        bot.emit("error", handleError(r));
    }
    await returnItems();
    // wait for last message
    await bot.waitForTicks(bot.waitTicks);
    if (!response_sent) {
        response_sent = true;
        res.json(bot.observe());
    }
    bot.removeListener("physicTick", onTick);

    async function evaluateCode(code, programs) {
        // Echo the code produced for players to see it. Don't echo when the bot code is already producing dialog or it will double echo
        try {
            await eval("(async () => {" + programs + "\n" + code + "})()");
            return "success";
        } catch (err) {
            return err;
        }
    }

    function onStuck(posThreshold) {
        const currentPos = bot.entity.position;
        bot.stuckPosList.push(currentPos);

        // Check if the list is full
        if (bot.stuckPosList.length === 5) {
            const oldestPos = bot.stuckPosList[0];
            const posDifference = currentPos.distanceTo(oldestPos);

            if (posDifference < posThreshold) {
                teleportBot(); // execute the function
            }

            // Remove the oldest time from the list
            bot.stuckPosList.shift();
        }
    }

    function teleportBot() {
        const blocks = bot.findBlocks({
            matching: (block) => {
                return block.type === 0;
            },
            maxDistance: 1,
            count: 27,
        });

        if (blocks) {
            // console.log(blocks.length);
            const randomIndex = Math.floor(Math.random() * blocks.length);
            const block = blocks[randomIndex];
            bot.chat(`/tp @s ${block.x} ${block.y} ${block.z}`);
        } else {
            bot.chat("/tp @s ~ ~1.25 ~");
        }
    }

    function returnItems() {
        bot.chat("/gamerule doTileDrops false");
        const crafting_table = bot.findBlock({
            matching: mcData.blocksByName.crafting_table.id,
            maxDistance: 128,
        });
        if (crafting_table) {
            bot.chat(
                `/setblock ${crafting_table.position.x} ${crafting_table.position.y} ${crafting_table.position.z} air destroy`
            );
            bot.chat("/give @s crafting_table");
        }
        const furnace = bot.findBlock({
            matching: mcData.blocksByName.furnace.id,
            maxDistance: 128,
        });
        if (furnace) {
            bot.chat(
                `/setblock ${furnace.position.x} ${furnace.position.y} ${furnace.position.z} air destroy`
            );
            bot.chat("/give @s furnace");
        }
        if (bot.inventoryUsed() >= 32) {
            // if chest is not in bot's inventory
            if (!bot.inventory.items().find((item) => item.name === "chest")) {
                bot.chat("/give @s chest");
            }
        }
        // if iron_pickaxe not in bot's inventory and bot.iron_pickaxe
        if (
            bot.iron_pickaxe &&
            !bot.inventory.items().find((item) => item.name === "iron_pickaxe")
        ) {
            bot.chat("/give @s iron_pickaxe");
        }
        bot.chat("/gamerule doTileDrops true");
    }

    function handleError(err) {
        let stack = err.stack;
        if (!stack) {
            return err;
        }
        console.log(stack);
        const final_line = stack.split("\n")[1];
        const regex = /<anonymous>:(\d+):\d+\)/;

        const programs_length = programs.split("\n").length;
        let match_line = null;
        for (const line of stack.split("\n")) {
            const match = regex.exec(line);
            if (match) {
                const line_num = parseInt(match[1]);
                if (line_num >= programs_length) {
                    match_line = line_num - programs_length;
                    break;
                }
            }
        }
        if (!match_line) {
            return err.message;
        }
        let f_line = final_line.match(
            /\((?<file>.*):(?<line>\d+):(?<pos>\d+)\)/
        );
        if (f_line && f_line.groups && fs.existsSync(f_line.groups.file)) {
            const { file, line, pos } = f_line.groups;
            const f = fs.readFileSync(file, "utf8").split("\n");
            // let filename = file.match(/(?<=node_modules\\)(.*)/)[1];
            let source = file + `:${line}\n${f[line - 1].trim()}\n `;

            const code_source =
                "at " +
                code.split("\n")[match_line - 1].trim() +
                " in your code";
            return source + err.message + "\n" + code_source;
        } else if (
            f_line &&
            f_line.groups &&
            f_line.groups.file.includes("<anonymous>")
        ) {
            const { file, line, pos } = f_line.groups;
            let source =
                "Your code" +
                `:${match_line}\n${code.split("\n")[match_line - 1].trim()}\n `;
            let code_source = "";
            if (line < programs_length) {
                source =
                    "In your program code: " +
                    programs.split("\n")[line - 1].trim() +
                    "\n";
                code_source = `at line ${match_line}:${code
                    .split("\n")
                    [match_line - 1].trim()} in your code`;
            }
            return source + err.message + "\n" + code_source;
        }
        return err.message;
    }
});

app.post("/stop", (req, res) => {
    bot.end();
    res.json({
        message: "Bot stopped",
    });
});

app.post("/pause", (req, res) => {
    if (!bot) {
        res.status(400).json({ error: "Bot not spawned" });
        return;
    }
    bot.chat("/pause");
    bot.waitForTicks(bot.waitTicks).then(() => {
        res.json({ message: "Success" });
    });
});

// Server listening to PORT 3000

const DEFAULT_PORT = 3001;
const PORT = process.argv[2] || DEFAULT_PORT;
app.listen(PORT, () => {
    console.log(`Server started on port ${PORT}`);
});


const puppeteer = require('puppeteer');
const path = require('path');


async function setupVisionCapture(bot) {
    const browser = await puppeteer.launch({ headless: false }); // Set headless to false for debugging
    const page = await browser.newPage();
    //await page.setViewport({ width: 320, height: 240 });
    console.log('Connecting to viewer...');
    // TODO: change to 3007 do not need to wait for viewer to load, then capture the screenshot
    await page.goto('http://localhost:3001'); // Connect to the viewer
    //await page.waitForSelector('#viewer-element');

    let lastCaptureTime = Date.now();
    const captureInterval = 1000 / 2 
    //16; // Approximately 62.5 ms for 16 fps

    // Ensure the logging folder exists
    const loggingFolder = path.resolve('/Users/daisysong/Desktop/CS194agent/Voyager_OAI/logs/visions');
    if (!fs.existsSync(loggingFolder)) {
        fs.mkdirSync(loggingFolder, { recursive: true });
    }

    // Maximum number of frames to keep
    const maxFrames = 1000; // Adjust this number based on your storage capacity
    const frameFiles = []; // Keep track of saved frame file names
    const sharp = require('sharp'); 

    async function captureAndSave() {
        try {
            console.log("Attempting to capture and save screenshot...");
            // Take screenshot and save to logging folder with compression
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-'); // Make filename compatible
            const screenshotFilename = `${bot.bot_Id}-screenshot-${timestamp}.jpg`;
            const screenshotPath = path.join(loggingFolder, screenshotFilename);

            await page.screenshot({
                path: screenshotPath,
                type: 'jpeg',
                quality: 30, // Adjust quality between 0-100# todo ask gpt to change to black and white
            });
            // new add
            // Convert the screenshot to grayscale using sharp
            //const grayscalePath = path.join(loggingFolder, `screenshot_bw-${timestamp}.jpg`);
            await sharp(screenshotPath)
                .grayscale() // Convert to grayscale
                .toFile(grayscalePath); // Save the grayscale image

            console.log(`Grayscale screenshot saved: ${grayscalePath}`);
            // new add

            // Collect metadata
            const metadata = {
                bot_id: bot.bot_id,
                mc_port: bot.mc_port,
                timestamp: new Date().toISOString(),
                position: bot.entity.position,
                orientation: bot.entity.yaw,
                // biome: bot.biome,
                // timeOfDay: bot.timeOfDay,
                image_path: screenshotPath,
                //image_path: grayscalePath,
                screenshotFilename: screenshotFilename,
                // inventory: bot.inventory.items().map((item) => ({
                //     name: item.name,
                //     count: item.count,
                // })),
            };

            // Save metadata asynchronously
            const metadataFilename = `${bot.bot_Id}-metadata-${timestamp}.json`;
            const metadataPath = path.join(loggingFolder, metadataFilename);

            fs.writeFile(metadataPath, JSON.stringify(metadata, null, 2), (err) => {
                if (err) {
                    console.error('Error saving metadata:', err);
                } else {
                    console.log(`Metadata saved: ${metadataPath}`);
                }
            });

            // Keep track of saved frames
            frameFiles.push({ screenshot: screenshotPath, metadata: metadataPath });

            // Manage disk space by deleting old frames
            // if (frameFiles.length > maxFrames) {
            //     const oldFrame = frameFiles.shift();
            //     fs.unlink(oldFrame.screenshot, (err) => {
            //         if (err) {
            //             console.error('Error deleting old screenshot:', err);
            //         } else {
            //             console.log(`Old screenshot deleted: ${oldFrame.screenshot}`);
            //         }
            //     });
            //     fs.unlink(oldFrame.metadata, (err) => {
            //         if (err) {
            //             console.error('Error deleting old metadata:', err);
            //         } else {
            //             console.log(`Old metadata deleted: ${oldFrame.metadata}`);
            //         }
            //     });
            // }

            console.log(`Screenshot saved: ${screenshotPath}`);

        } catch (error) {
            console.error('Error capturing and saving screenshot:', error);
        }
    }

    // For testing, use setInterval instead of bot.on('move')
    const captureLoop = setInterval(captureAndSave, captureInterval);

    // Clean up function
    const cleanup = async () => {
        clearInterval(captureLoop);
        await browser.close();
    };

    // Add cleanup to bot's end event
    bot.once('end', cleanup);

    // Return cleanup function in case you need to stop early
    return cleanup;
}

//module.exports = setupVisionCapture;


// Sets up an Express server with endpoints to start, stop, and control a Minecraft bot using Mineflayer.
// Initializes the bot and loads various plugins and observations.
// Sets up a viewer using prismarine-viewer to visualize the bot in a web browser.
// Implements a setupVisionCapture function that uses Puppeteer to capture screenshots and metadata from the viewer at regular intervals.

// Based on the code provided and common pitfalls, here are possible areas where issues might occur:

// Asynchronous Initialization and Timing Issues
// Scope and Variable Accessibility
// Error Handling and Logging
// Puppeteer and Viewer Interaction
// Resource Cleanup