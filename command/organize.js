let fs = require("fs");
let path = require("path");

let types = {
    media: ["mp4", "mkv", "jpg", "png", "jpeg", "PNG", "JPEG", "JPG", "gif"],
    archives: ['zip', '7z', 'rar', 'tar', 'gz', 'ar', 'iso', "xz"],
    documents: ['docx', 'doc', 'pdf', 'xlsx', 'xls', 'odt', 'ods', 'odp', 'odg', 'odf', 'txt', 'ps', 'tex', 'pptx'],
    app: ['exe', 'dmg', 'pkg', "deb"]
}


function organizeOutput(src){
    let newFolder;
    if(src == undefined){
        console.log("Please provide the path.");
    }else{
        let isPresent = fs.existsSync(src);
        if(isPresent){
            newFolder = path.join(src, "OrganizedFile");
            if(fs.existsSync(newFolder) == false){
                fs.mkdirSync(newFolder);
            }

        }else{
            console.log("Please provide the correct path");
        }
    }
    organizer(src,newFolder);
}


function organizer(src,dest){
    let content = fs.readdirSync(src);
     
     //console.log( content);
    for(let i = 0; i < content.length; i++){
        let contentAddress = path.join(src, content[i]);
        let isFile = fs.lstatSync(contentAddress).isFile(contentAddress);
        if(isFile){
            let categoryOfFile = categorycheck(content[i]);
            putFile(contentAddress,dest,categoryOfFile);           
        }
    }
    
}

function categorycheck(name){
    let extension = path.extname(name);
    extension = extension.slice(1);
    for(let key in types){
        let currentKeyArray = types[key];
        for(let j = 0; j < currentKeyArray.length; j++){
            if(extension == currentKeyArray[j]){
                return key;
            }
        }
    }
    return "Others";
}

function putFile(src,destination,category){
    let fileCategoryPath = path.join(destination, category);
    if(fs.existsSync(fileCategoryPath) == false){
            fs.mkdirSync(fileCategoryPath);
    }
    let tobeCopied = path.basename(src);
    let destPath = path.join(fileCategoryPath, tobeCopied);
    fs.copyFileSync(src, destPath); 
    //fs.unlinkSync(src);   
}

module.exports = {
    organizefunction: organizeOutput
}