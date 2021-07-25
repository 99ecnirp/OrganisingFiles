let fs = require("fs");
let path = require("path");

function treeOutput(src){
    if(src == undefined){
    console.log("Please provide the path.");
    }else
    {
        let isPresent = fs.existsSync(src);
        if(isPresent)
        {
           treeMaker(src,"  ")
        }else
          {
              console.log("Please provide the correct path");
          }
  }
}
function treeMaker(src,indent){
    let isFile = fs.lstatSync(src).isFile();
    if(isFile == true){
        let fileName = path.basename(src);
        console.log(indent + "|---" + fileName);
    }else{
        let folderName = path.basename(src);
        console.log(indent + "|___" + folderName);
        let content = fs.readdirSync(src);
        for(let i = 0; i < content.length; i++){
           let contentAddress = path.join(src, content[i]);
           treeMaker(contentAddress,indent+"\t");
        }
    }
}


module.exports = {
   treefunction: treeOutput
}