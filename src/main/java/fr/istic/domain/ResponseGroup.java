package fr.istic.domain;

import io.quarkus.hibernate.orm.panache.PanacheEntityBase;
import io.quarkus.hibernate.orm.panache.PanacheQuery;
import jakarta.json.bind.annotation.JsonbTransient;
import io.quarkus.runtime.annotations.RegisterForReflection;

import jakarta.persistence.*;
import java.io.Serializable;
import java.util.List;
import java.util.Set;

/**
 * A Response Group.
 */
@Entity
@Table(name = "response_group")
@RegisterForReflection
public class ResponseGroup extends PanacheEntityBase implements Serializable {

    private static final long serialVersionUID = 1L;

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    public Long id;

    @ManyToOne
    @JoinColumn(name = "question_id")
    @JsonbTransient
    public Question question;

    @Column(name="prediction_ids")
    public Long[] predictionIds;

    @Column(name="average_embedding")
    public Double[] averageEmbedding;



    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ResponseGroup)) {
            return false;
        }
        return id != null && id.equals(((ResponseGroup) o).id);
    }

    @Override
    public int hashCode() {
        return 31;
    }

    @Override
    public String toString() {
        return "ResponseGroup{" +
            "id=" + id +
            ", question='" + question + "'" +
            ", predictionIds='" + predictionIds + "'" +
            ", averageEmbedding='" + averageEmbedding + "'" +
            "}";
    }

    public ResponseGroup update() {
        return update(this);
    }

    public ResponseGroup persistOrUpdate() {
        return persistOrUpdate(this);
    }

    public static ResponseGroup update(ResponseGroup responseGroup) {
        if (responseGroup == null) {
            throw new IllegalArgumentException("responseGroup can't be null");
        }
        var entity = ResponseGroup.<ResponseGroup>findById(responseGroup.id);
        if (entity != null) {
            entity.question = responseGroup.question;
            entity.predictionIds = responseGroup.predictionIds;
            entity.averageEmbedding = responseGroup.averageEmbedding;
        }
        return entity;
    }

    public static ResponseGroup persistOrUpdate(ResponseGroup responseGroup) {
        if (responseGroup == null) {
            throw new IllegalArgumentException("responseGroup can't be null");
        }
        if (responseGroup.id == null) {
            persist(responseGroup);
            return responseGroup;
        } else {
            return update(responseGroup);
        }
    }


    public static PanacheQuery<ResponseGroup> findByQuestionId(long qid) {
        return find("select rg from ResponseGroup rg where rg.question.id = ?1", qid);
    }
    
    public static long deleteByQIds(Set<Long> qids) {
        return delete("delete from ResponseGroup rg where rg.question.id in ?1", qids);
    }
    
    public static long deleteByQId(Long qid) {
        return delete("delete from ResponseGroup rg where rg.question.id = ?1", qid);
    }
    
    public static PanacheQuery<ResponseGroup> canAccess(long responseGroupId, String login) {
        return find("select rg from ResponseGroup rg join rg.question.exam.course.rgofs as u where rg.id = ?1 and u.login = ?2", responseGroupId, login);
    }
    
    public static PanacheQuery<ResponseGroup> findByQuestion(Long questionId) {
        return find("select rg from ResponseGroup rg where rg.question.id = ?1", questionId);
    }
    
    public static PanacheQuery<ResponseGroup> findByPrediction(Long predictionId) {
        return find("select rg from ResponseGroup rg where ?1 member of rg.predictionIds", predictionId);
    }
}
