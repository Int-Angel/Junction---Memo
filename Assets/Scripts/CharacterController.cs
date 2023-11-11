using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum AnimationState
{
    Idle = 1,
    Punch = 2,
    Run = 3,
    Jump = 4,
    Sleep = 5,
    Waving = 6
}

public class CharacterController : MonoBehaviour
{
    Animator animator;
    public float TestSpeed = 1;
    public float animation = 1;

    // Start is called before the first frame update
    void Start()
    {
        animator = GetComponent<Animator>();
        // (int)AnimationState.Idle
        animator.SetFloat("State", animation);
        animator.speed = TestSpeed;
    }

    // Update is called once per frame
    void Update()
    {
        animator.SetFloat("State", animation);
        animator.speed = TestSpeed;
    }
}
