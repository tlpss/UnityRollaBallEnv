using System;
using Unity.MLAgents;
using UnityEngine;
using Random = UnityEngine.Random;
using Vector3 = UnityEngine.Vector3;

public class EnvironmentChannel
/*
 * ML-Agents Environment Parameters SideChannel Endpoint
 */

{
    // Singleton Construction
    private static EnvironmentChannel _instance = new EnvironmentChannel();
    private EnvironmentChannel()
    {
    }

    public static EnvironmentChannel Instance => _instance;
    // Environment Parameters are delivered to Academy by Environments SideChannel
    private readonly EnvironmentParameters _envParameters = Academy.Instance.EnvironmentParameters;    

    public Boolean Initialized 
    {
        get =>_envParameters.GetWithDefault("initialized", float.NaN).Equals(1.0f);
    }
    
    /* Actual Environment State */
    
    // Position of the Target 
    public Vector3 TargetPosition
    {
        get  
        {
            Vector3 targetPosition = new Vector3(0, 0, 0);
            targetPosition.y = 0.2f;// above floor
            targetPosition.x = _envParameters.GetWithDefault("target_position_x", float.NaN);
            targetPosition.z = _envParameters.GetWithDefault("target_position_z", float.NaN);
            
            // handle NaNs to allow for gameplay without Sidechannel
            // uniform randomization
            if  (float.IsNaN(targetPosition.x) || float.IsNaN(targetPosition.z))
            {
                Debug.LogError("TargetPosition was not set by SideChannel, although it was requested by the Agent");
            }
            return targetPosition;
        }
    }
    
    
    
}