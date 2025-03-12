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
@Table(name = "prediction")
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

    @Column(name="predictionIds")
    public double[] predictionIds;

    @Column(name="averageEmbedding")
    public double[] averageEmbedding;



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
        return find("select responseGroup from ResponseGroup responseGroup where responseGroup.question.id = ?1", qid);
    }

    public static long deleteByQIds(Set<Long> qids) {
        return delete("delete from ResponseGroup pr where pr.question.id in ?1", qids);
    }

    public static long deleteByQId(Long qid) {
        return delete("delete from ResponseGroup pr where pr.question.id = ?1", qid);
    }


    public static PanacheQuery<ResponseGroup> canAccess(long responseGroupId, String login) {
        return find("select pr from ResponseGroup pr join pr.question.exam.course.profs as u where pr.id = ?1 and u.login = ?2", responseGroupId, login);
    }

    public static PanacheQuery<ResponseGroup> findResponseGroupWithoutStudentResponse(List<Long> responseGroupIds) {
        return find("select pr from ResponseGroup pr where pr.id in ?1 and pr.question.studentResponse is null", responseGroupIds);
    }
}
