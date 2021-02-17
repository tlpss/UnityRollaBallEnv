using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;

public class RollerAgent: Agent
{
    // NB: rigidbody -> physics simulation, transform -> position,orientation,size
    public Transform Target; // TF of the target object 

    private Rigidbody rAgent; // rigid body attached to agent
    public Vector3 initialAgentPose = new Vector3(0, 0.5f, 0);

    public float forceMultiplier = 10;

    // Start is called before the first frame update
    void Start()
    {
        // check behaviour sizes
        BehaviorParameters behaviorParameters = this.gameObject.GetComponent<BehaviorParameters>();
        Debug.Log("size of continuous observation vector = " + behaviorParameters.BrainParameters.VectorObservationSize);
        Debug.Log("size of continuous action vector = " + behaviorParameters.BrainParameters.ActionSpec.NumContinuousActions);

        rAgent = this.GetComponent<Rigidbody>(); // get rigid body attached to Agent gameobject
    }

    // gets called on begin of each episode
    // here the env is typically randomized
    public override void OnEpisodeBegin()
    {
        // test Sidechannel
        var envParameters = Academy.Instance.EnvironmentParameters;
        float targetX = envParameters.GetWithDefault("target_x", 0.0f);
        Debug.Log("received targetX = " + targetX );
        
        
        
        // bring agent momentum to zero 
        this.rAgent.angularVelocity = Vector3.zero;
        this.rAgent.velocity = Vector3.zero;
        // set initial position 
        this.transform.localPosition = initialAgentPose;
        // set target position
        Target.localPosition = new Vector3(Random.value * 8 - 4,
            0.5f, 
            Random.value * 8 - 4);
        
    }
    // create observations
    public override void CollectObservations(VectorSensor sensor)
    {
        /* 8D vector observation */
        // target pose
        sensor.AddObservation(Target.localPosition);
        // agent pose
        sensor.AddObservation(this.transform.localPosition);
        // agent velocity
        sensor.AddObservation(rAgent.velocity.x);
        sensor.AddObservation(rAgent.velocity.z);
    }
    //receives actions and assigns rewards
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Actions, size = 2
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rAgent.AddForce(controlSignal * forceMultiplier);

        // Rewards
        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        // Reached target
        if (distanceToTarget < 1.42f)
        {
            Debug.Log("reached");
            SetReward(1.0f);
            EndEpisode();
        }

        // Fell off platform
        else if (this.transform.localPosition.y < -0.1f)
        {
            Debug.Log("fell");
            EndEpisode();
        }
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;

        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");

    }
}
