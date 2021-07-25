let fs = require("fs");
let path = require("path");

let inputArr = process.argv.slice(2);
let inputCommand = inputArr[0];
let inputPath = inputArr[1];
//console.log(inputCommand);

let helpObj = require("./command/help");
let treeObj = require("./command/tree");
let organizeObj = require("./command/organize");


switch(inputCommand){
    case "tree":
        treeObj.treefunction(inputPath);
        break;
    case "organize":
        organizeObj.organizefunction(inputPath);
        break;
    case "help":
        helpObj.helpfunction(inputPath);
        break;
    default:
        console.log("Wrong input");            
}
