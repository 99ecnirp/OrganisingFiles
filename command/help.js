function helpOutput(src){
    console.log(`List of All Commands
                      1.Node main.js tree "path" ${src} 
                      2.Node main.js organize "path"  + ${src}
                      3.Node main.js help`);
}

module.exports = {
   helpfunction: helpOutput
}