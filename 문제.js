
var array = [{ index : 1, cd : 'a', up_cd : 'z'}
                , { index : 2, cd : 'b', up_cd : 'a'}
                    , { index : 3, cd : 'a', up_cd : 'b'}
                    , { index : 4, cd : 'e', up_cd : 'b'}
                , { index : 5, cd : 'c', up_cd : 'a'}
                    , { index : 6, cd : 'f', up_cd : 'c'}
                    , { index : 7, cd : 'g', up_cd : 'c'}
                , { index : 8, cd : 'c' , up_cd : 'a'} ]
var map = {}
var visited = {}

for(var i = 0 ; i < array.length ; i++){

    if(array[i].up_cd in map){
        map[array[i].up_cd].push([array[i].cd,array[i].index]);
    }else{
        map[array[i].up_cd] = [[array[i].cd,array[i].index]]   
    }
    
}
var resultArr = [];
var que = [['a',1]]


while(que.length > 0){
    
    var v = que.shift()
    var up_cd = v[0]
    var index = v[1]
    visited[up_cd] = true
    var nextArr = map[up_cd]

    for(var row in nextArr){
        if(resultArr.indexOf(nextArr[row][1]) == -1 ){
            que.push(nextArr[row])
            resultArr.push(nextArr[row][1])
        }
    }
}

console.log(resultArr)


                