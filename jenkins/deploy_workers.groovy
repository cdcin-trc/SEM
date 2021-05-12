/**
This script will deploy CDCIN Workers to Jenkins Slaves with the 're' label
This script requires the following Jenkins plugins:
-Pipeline: Utility Steps
*/

//Load shared pipeline utility library
@Library('cditma-utils')
import cditma.Utils

def utils = new Utils(this);

def re_nodes = nodesByLabel("deploy_re")
final STASH_NAME = "worker_stash"

//Checkout and stash re source archive (stored on Master's local git repo)
stage('Checkout'){
    node('master'){
        checkout scm
        
        dir("deployment"){       
            stash includes: "**", name: STASH_NAME
        }
    }
}

//Construct build map for all nodes which should build
def step_build = [:]
for(n in re_nodes){
    def node_name = n
    if(node_name == ""){
        node_name = "master"
    }
    step_build[node_name] = {
        node(node_name){
            dir("${HOME}/cdcin"){
                deleteDir()
            }
            dir("${HOME}/cdcin"){         
                unstash STASH_NAME
                dir("build"){
                    def result = utils.buildProject("Ninja", "")
                    if(!result){
                        error('Failed to compile')
                    }
                }
            }
        }
    }
}

//Build re on all re nodes
stage('Build'){
    parallel step_build
}
